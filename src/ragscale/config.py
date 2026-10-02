"""Load and validate a ragscale.yaml into one RunConfig."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ragscale.adapters import provenance_hash, resolve_adapter
from ragscale.cache import read_jsonl
from ragscale.scoring import resolve_scorer


@dataclass
class ReaderSpec:
    key: str
    name: str
    family: str
    callable: Any
    provenance: dict[str, Any]

    @property
    def provenance_hash(self) -> str:
        return provenance_hash(self.provenance)


@dataclass
class RunConfig:
    config_path: Path
    dataset_name: str
    dataset_path: Path
    dataset_description: str | None
    dataset_hints: dict[str, Any]
    hint_fields: list[str]
    minimum_slice_rows: int
    cases: list[dict[str, Any]]
    raw_policy_id: str
    compressor_id: str
    compressor: Any
    compressor_provenance: dict[str, Any]
    readers: dict[str, ReaderSpec]
    current_key: str
    candidate_key: str
    metrics: list[tuple[str, Any]]
    seed: int
    bootstrap_draws: int
    minimum_raw_gap: float
    retention_alert_threshold: float
    workers: int
    store_artifact_text: bool
    reuse_artifacts: bool
    reuse_answers: bool
    fail_on_outcomes: list[str] = field(default_factory=list)
    output_dir: Path = Path("ragscale-audit")

    @property
    def compressor_provenance_hash(self) -> str:
        return provenance_hash(self.compressor_provenance)

    @property
    def current(self) -> ReaderSpec:
        return self.readers[self.current_key]

    @property
    def candidate(self) -> ReaderSpec:
        return self.readers[self.candidate_key]


def read_yaml(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if int(config.get("version", 1)) != 1:
        raise ValueError("Only ragscale configuration version 1 is supported")
    return config


def canonical_evidence(value: Any) -> str:
    """Join a candidate pool into the raw evidence string shown to readers."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        documents = []
        for document in value:
            if isinstance(document, str):
                documents.append(document)
            elif isinstance(document, dict):
                text = document.get("text", document.get("content", document.get("page_content")))
                documents.append(str(text) if text is not None else json.dumps(document, sort_keys=True))
            else:
                documents.append(str(document))
        return "\n\n".join(documents)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def validate_case(case: dict[str, Any], line_number: int) -> None:
    missing = {"example_id", "question", "candidate_pool", "reference_answers"} - set(case)
    if missing:
        raise ValueError(f"Dataset example {line_number} is missing: {', '.join(sorted(missing))}")
    references = case["reference_answers"]
    if isinstance(references, str):
        case["reference_answers"] = [references]
    elif not isinstance(references, list) or not references:
        raise ValueError(f"Dataset example {line_number} needs one or more reference_answers")


def hint_value(case: dict[str, Any], field_name: str, default: str = "unspecified") -> str:
    if field_name in case and case[field_name] not in (None, ""):
        value = case[field_name]
    else:
        value = (case.get("hints") or {}).get(field_name, default)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _unit_interval(run: dict[str, Any], name: str, default: float) -> float:
    value = float(run.get(name, default))
    if not 0 <= value <= 1:
        raise ValueError(f"run.{name} must lie in [0, 1]")
    return value


def load_config(config_path: str | Path, *, output_dir: str | Path | None = None) -> RunConfig:
    path = Path(config_path).resolve()
    config = read_yaml(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))

    dataset_config = config.get("dataset") or {}
    if not dataset_config.get("path"):
        raise ValueError("dataset.path is required")
    dataset_path = Path(dataset_config["path"])
    if not dataset_path.is_absolute():
        dataset_path = path.parent / dataset_path
    cases = read_jsonl(dataset_path)
    for index, case in enumerate(cases, 1):
        validate_case(case, index)
    case_ids = [str(case["example_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Dataset example_id values must be unique")
    minimum_slice_rows = int(dataset_config.get("minimum_slice_rows", 10))
    if minimum_slice_rows <= 0:
        raise ValueError("dataset.minimum_slice_rows must be positive")

    pipeline = config.get("pipeline") or {}
    compressor_config = pipeline.get("compressor")
    if not compressor_config:
        raise ValueError("pipeline.compressor is required")
    if isinstance(compressor_config, str):
        compressor_id = compressor_config
    else:
        compressor_id = compressor_config.get("id") or compressor_config.get("callable")
        if not compressor_id and compressor_config.get("type") == "openai_compatible":
            compressor_id = f"openai-compatible:{compressor_config.get('model', 'unspecified')}"
    compressor, compressor_provenance = resolve_adapter(
        compressor_config, role="compressor", base_dir=path.parent
    )

    readers_config = config.get("readers") or {}
    if len(readers_config) < 2:
        raise ValueError("Configure at least two readers")
    readers = {}
    for key, value in readers_config.items():
        value = {"callable": value} if isinstance(value, str) else dict(value)
        adapter, adapter_provenance = resolve_adapter(value, role="reader", base_dir=path.parent)
        readers[key] = ReaderSpec(
            key=key, name=str(value.get("name", key)),
            family=str(value.get("family", "unspecified")),
            callable=adapter, provenance=adapter_provenance,
        )
    names = [reader.name for reader in readers.values()]
    if len(names) != len(set(names)):
        raise ValueError("Configured reader names must be unique")
    comparison = config.get("comparison") or {}
    current_key = str(comparison.get("current", "current"))
    candidate_key = str(comparison.get("candidate", "candidate"))
    if current_key not in readers or candidate_key not in readers:
        raise ValueError("comparison.current and comparison.candidate must name configured readers")

    scoring = config.get("scoring") or {}
    metrics = [
        resolve_scorer(spec, path.parent)
        for spec in scoring.get("metrics") or ["exact_match", "token_f1"]
    ]
    metric_names = [name for name, _ in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("Configured metric names must be unique")

    run = config.get("run") or {}
    bootstrap_draws = int(run.get("bootstrap_draws", 2_000))
    if bootstrap_draws <= 0:
        raise ValueError("run.bootstrap_draws must be positive")
    workers = int(run.get("workers", 1))
    if workers <= 0:
        raise ValueError("run.workers must be positive")
    output = Path(output_dir or run.get("output_dir", "ragscale-audit"))
    if not output.is_absolute():
        output = path.parent / output

    return RunConfig(
        config_path=path,
        dataset_name=str(dataset_config.get("name", "user-dataset")),
        dataset_path=dataset_path,
        dataset_description=dataset_config.get("description"),
        dataset_hints=dict(dataset_config.get("hints") or {}),
        hint_fields=[str(name) for name in dataset_config.get("hint_fields", ["question_type"])],
        minimum_slice_rows=minimum_slice_rows,
        cases=cases,
        raw_policy_id=str(pipeline.get("raw_policy_id", "raw-evidence-v1")),
        compressor_id=str(compressor_id),
        compressor=compressor,
        compressor_provenance=compressor_provenance,
        readers=readers,
        current_key=current_key,
        candidate_key=candidate_key,
        metrics=metrics,
        seed=int(run.get("seed", 20260821)),
        bootstrap_draws=bootstrap_draws,
        minimum_raw_gap=_unit_interval(run, "minimum_raw_gap", 0.05),
        retention_alert_threshold=_unit_interval(run, "retention_alert_threshold", 0.75),
        workers=workers,
        store_artifact_text=bool(run.get("store_artifact_text", True)),
        reuse_artifacts=bool(run.get("reuse_artifacts", True)),
        reuse_answers=bool(run.get("reuse_answers", True)),
        fail_on_outcomes=[str(value) for value in run.get("fail_on_outcomes", [])],
        output_dir=output,
    )
