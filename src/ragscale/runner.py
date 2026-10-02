"""Run one YAML-configured audit: load, compile, replay, analyze, report."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ragscale.adapters import AdapterOutput  # noqa: F401  (public re-export)
from ragscale.analysis import analyze_metric
from ragscale.cache import CallCache
from ragscale.config import RunConfig, load_config
from ragscale.execute import compile_artifacts, replay_readers
from ragscale.presets import DEMO_DATASET, bundled_data_path, write_starter_project  # noqa: F401
from ragscale.reports import write_outputs
from ragscale.scoring import JudgeScorer


def run_config(config_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run a YAML-configured compile-once reader comparison and write its outputs."""
    config = load_config(config_path, output_dir=output_dir)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cache = CallCache(output / "call_cache.jsonl", enabled=config.reuse_answers)

    compiled = compile_artifacts(config, cache, output / "artifact_manifest.jsonl")
    replay = replay_readers(config, compiled, cache)
    frame = replay.frame

    metric_results = [
        analyze_metric(
            frame, metric, config.current.name, config.candidate.name,
            bootstrap_draws=config.bootstrap_draws, seed=config.seed + index,
            minimum_gap=config.minimum_raw_gap,
            retention_alert_threshold=config.retention_alert_threshold,
            hint_fields=config.hint_fields, minimum_slice_rows=config.minimum_slice_rows,
        )
        for index, (metric, _) in enumerate(config.metrics)
    ]
    summary = _summary(config, frame, metric_results, replay.judge_calls, replay.judge_cost_usd, cache)
    write_outputs(summary, frame, output)
    return summary


def _summary(config: RunConfig, frame, metric_results, judge_calls: int, judge_cost: float,
             cache: CallCache) -> dict[str, Any]:
    reader_rows = frame.drop_duplicates(["example_id", "reader"])
    compressor_rows = frame.drop_duplicates(["example_id"])
    reported_cost = float(
        reader_rows[["raw_cost_usd", "compressed_cost_usd"]].fillna(0).sum().sum()
        + compressor_rows["compressor_cost_usd"].fillna(0).sum()
        + judge_cost
    )
    return {
        "schema_version": 1,
        "dataset": {
            "name": config.dataset_name,
            "path": str(config.dataset_path),
            "rows": len(config.cases),
            "description": config.dataset_description,
            "hints": config.dataset_hints,
            "hint_fields": config.hint_fields,
        },
        "comparison": {"current": config.current.name, "candidate": config.candidate.name},
        "release_gate": {
            "retention_alert_threshold": config.retention_alert_threshold,
            "fail_on_outcomes": config.fail_on_outcomes,
        },
        "metrics": [asdict(result) for result in metric_results],
        "provenance": {
            "candidate_pools": int(frame.candidate_pool_id.nunique()),
            "compressed_artifacts": len(config.cases),
            "unique_compressed_artifact_hashes": int(frame.compressed_artifact_hash.nunique()),
            "exact_reader_footprints": all(
                group.example_id.nunique() == len(config.cases)
                for _, group in frame.groupby(["metric", "reader"])
            ),
            "compressor_id": config.compressor_id,
            "adapter_provenance": {
                "compressor": config.compressor_provenance,
                "readers": {key: reader.provenance for key, reader in config.readers.items()},
                "judges": {
                    name: scorer.provenance for name, scorer in config.metrics
                    if isinstance(scorer, JudgeScorer)
                },
            },
            "raw_policy_id": config.raw_policy_id,
            "artifact_manifest": str(config.output_dir / "artifact_manifest.jsonl"),
            "artifacts_reused": int(compressor_rows.artifact_reused.sum()),
        },
        "resource_summary": {
            "mean_compressor_latency_ms": float(compressor_rows.compressor_latency_ms.mean()),
            "mean_raw_reader_latency_ms": float(reader_rows.raw_latency_ms.mean()),
            "mean_compressed_reader_latency_ms": float(reader_rows.compressed_latency_ms.mean()),
            "reported_cost_usd": reported_cost,
            "judge_cost_usd": judge_cost,
            "judge_calls_this_run": judge_calls,
            "compressor_calls_this_run": int((~compressor_rows.artifact_reused).sum()),
            "reader_calls_this_run": int(
                (~reader_rows.raw_answer_reused).sum() + (~reader_rows.compressed_answer_reused).sum()
            ),
            "workers": config.workers,
        },
        "run": {
            "seed": config.seed,
            "bootstrap_draws": config.bootstrap_draws,
            "minimum_raw_gap": config.minimum_raw_gap,
            "workers": config.workers,
            "reuse_artifacts": config.reuse_artifacts,
            "reuse_answers": config.reuse_answers,
            "call_cache": str(cache.path) if config.reuse_answers else None,
        },
        "scope": "Observed readers and stored compile-once artifacts only; no inference to unseen readers or future artifact draws.",
        "output_dir": str(config.output_dir),
    }
