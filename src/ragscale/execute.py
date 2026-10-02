"""Compile each item once, then replay every reader under raw and compressed evidence."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ragscale.adapters import AdapterOutput, invoke_adapter
from ragscale.cache import CallCache, digest, json_safe, read_jsonl, run_parallel
from ragscale.config import RunConfig, canonical_evidence, hint_value
from ragscale.scoring import JudgeScorer, score_answer


@dataclass
class CompiledCase:
    case: dict[str, Any]
    raw_evidence: str
    candidate_pool_id: str
    metadata: dict[str, Any]
    compressed: AdapterOutput
    compressed_hash: str
    artifact_reused: bool


def _case_metadata(config: RunConfig, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": str(case["example_id"]),
        "dataset": config.dataset_name,
        "question_type": hint_value(case, "question_type"),
        "hints": dict(case.get("hints") or {}),
        "tags": case.get("tags", []),
    }


def _reused(checkpoint: dict[str, Any]) -> AdapterOutput:
    metadata = dict(checkpoint.get("metadata") or {})
    metadata["measured_latency_ms"] = 0.0
    metadata["reused"] = True
    return AdapterOutput(
        text=str(checkpoint["text"]), tokens=checkpoint.get("tokens"), cost_usd=0.0, metadata=metadata,
    )


def _from_manifest(config: RunConfig, case: dict[str, Any], row: dict[str, Any],
                   candidate_pool_id: str, raw_evidence: str) -> AdapterOutput:
    """Reuse a manifest artifact, refusing if anything upstream of it changed."""
    example_id = case["example_id"]
    if row.get("candidate_pool_id") != candidate_pool_id:
        raise ValueError(f"Cached candidate pool differs for {example_id}")
    if row.get("raw_evidence_hash") != digest(raw_evidence):
        raise ValueError(f"Cached raw evidence differs for {example_id}")
    if row.get("compressor_id") != config.compressor_id:
        raise ValueError(f"Cached compressor identifier differs for {example_id}")
    adapter_hash = row.get("compressor_adapter_hash")
    if adapter_hash and adapter_hash != config.compressor_provenance_hash:
        raise ValueError(f"Cached compressor adapter configuration differs for {example_id}")
    if row.get("compressed_evidence") is None:
        raise ValueError(
            f"Cached artifact text is unavailable for {example_id}; "
            "set reuse_artifacts: false to compile again"
        )
    compressed = AdapterOutput(
        text=str(row["compressed_evidence"]), tokens=row.get("compressor_tokens"),
        cost_usd=0.0, metadata={"measured_latency_ms": 0.0},
    )
    if digest(compressed.text) != row.get("compressed_artifact_hash"):
        raise ValueError(f"Cached compressed artifact hash differs for {example_id}")
    return compressed


def compile_artifacts(config: RunConfig, cache: CallCache, manifest_path: Path) -> list[CompiledCase]:
    """Run the compressor once per item and write the artifact manifest."""
    manifest_rows = {}
    if config.reuse_artifacts and manifest_path.exists():
        for row in read_jsonl(manifest_path):
            manifest_rows[str(row["example_id"])] = row

    def compile_case(case: dict[str, Any]) -> CompiledCase:
        raw_evidence = canonical_evidence(case["candidate_pool"])
        candidate_pool_id = str(case.get("candidate_pool_id") or digest(raw_evidence))
        metadata = _case_metadata(config, case)
        manifest_row = manifest_rows.get(str(case["example_id"]))
        if manifest_row:
            compressed = _from_manifest(config, case, manifest_row, candidate_pool_id, raw_evidence)
            reused = True
        else:
            key = CallCache.key(
                "compressor", case["example_id"], digest(raw_evidence),
                config.compressor_id, config.compressor_provenance_hash,
            )
            checkpoint = cache.get(key)
            if checkpoint is not None:
                compressed = _reused(checkpoint)
                reused = True
            else:
                compressed = invoke_adapter(
                    config.compressor, question=str(case["question"]),
                    documents=case["candidate_pool"], evidence=raw_evidence,
                    metadata=metadata, case=case,
                )
                cache.put(key, {
                    "text": compressed.text, "tokens": compressed.tokens,
                    "cost_usd": compressed.cost_usd, "metadata": compressed.metadata,
                })
                reused = False
        return CompiledCase(
            case=case, raw_evidence=raw_evidence, candidate_pool_id=candidate_pool_id,
            metadata=metadata, compressed=compressed, compressed_hash=digest(compressed.text),
            artifact_reused=reused,
        )

    compiled = run_parallel(compile_case, config.cases, config.workers)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in compiled:
            row = {
                "example_id": str(item.case["example_id"]),
                "candidate_pool_id": item.candidate_pool_id,
                "raw_evidence_hash": digest(item.raw_evidence),
                "compressed_artifact_hash": item.compressed_hash,
                "compressor_id": config.compressor_id,
                "compressor_adapter_hash": config.compressor_provenance_hash,
                "question_type": item.metadata["question_type"],
                "compressed_evidence": item.compressed.text if config.store_artifact_text else None,
                "compressor_tokens": item.compressed.tokens,
                "compressor_metadata": item.compressed.metadata,
                "compilation_latency_ms": item.compressed.metadata["measured_latency_ms"],
                "artifact_reused": item.artifact_reused,
            }
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
    return compiled


@dataclass
class ReplayResult:
    frame: pd.DataFrame
    judge_calls: int
    judge_cost_usd: float


def replay_readers(config: RunConfig, compiled: list[CompiledCase], cache: CallCache) -> ReplayResult:
    """Ask every reader for raw and compressed answers, score them, and return one row per metric."""
    judge_lock = threading.Lock()
    judge_usage = {"calls": 0, "cost_usd": 0.0}
    judge_hashes = {
        name: scorer.provenance_hash for name, scorer in config.metrics
        if isinstance(scorer, JudgeScorer)
    }

    def answer(reader_key: str, item: CompiledCase, condition: str) -> tuple[AdapterOutput, bool]:
        reader = config.readers[reader_key]
        evidence = item.raw_evidence if condition == "raw" else item.compressed.text
        key = CallCache.key(
            "reader", item.case["example_id"], reader.name, condition,
            digest(evidence), reader.provenance_hash,
        )
        checkpoint = cache.get(key)
        if checkpoint is not None:
            return _reused(checkpoint), True
        output = invoke_adapter(
            reader.callable, question=str(item.case["question"]), evidence=evidence,
            metadata=item.metadata, case=item.case, condition=condition,
        )
        cache.put(key, {
            "text": output.text, "tokens": output.tokens,
            "cost_usd": output.cost_usd, "metadata": output.metadata,
        })
        return output, False

    def score(metric: str, scorer: Any, prediction: str, item: CompiledCase,
              reader_key: str, condition: str) -> tuple[float, bool]:
        case = item.case
        if not isinstance(scorer, JudgeScorer):
            return score_answer(scorer, prediction, case["reference_answers"], case), False
        key = CallCache.key(
            "judge", metric, case["example_id"], config.readers[reader_key].name, condition,
            digest(prediction), judge_hashes[metric],
        )
        checkpoint = cache.get(key)
        if checkpoint is not None:
            return float(checkpoint["score"]), True
        result = scorer.adapter.judge(prediction, case["reference_answers"], case)
        cache.put(key, result)
        with judge_lock:
            judge_usage["calls"] += 1
            judge_usage["cost_usd"] += float(result.get("cost_usd") or 0.0)
        return float(result["score"]), False

    def replay(task: tuple[CompiledCase, str]) -> list[dict[str, Any]]:
        item, reader_key = task
        case, reader = item.case, config.readers[reader_key]
        raw_answer, raw_reused = answer(reader_key, item, "raw")
        compressed_answer, compressed_reused = answer(reader_key, item, "compressed")
        rows = []
        for metric, scorer in config.metrics:
            raw_score, raw_judged = score(metric, scorer, raw_answer.text, item, reader_key, "raw")
            compressed_score, compressed_judged = score(
                metric, scorer, compressed_answer.text, item, reader_key, "compressed"
            )
            row = {
                "example_id": str(case["example_id"]),
                "reader": reader.name,
                "reader_key": reader_key,
                "reader_family": reader.family,
                "raw_score": raw_score,
                "compressed_score": compressed_score,
                "candidate_pool_id": item.candidate_pool_id,
                "compressed_artifact_hash": item.compressed_hash,
                "compressor_id": config.compressor_id,
                "raw_policy_id": config.raw_policy_id,
                "metric": metric,
                "dataset": item.metadata["dataset"],
                "question": str(case["question"]),
                "reference_answers_json": json.dumps(case["reference_answers"], ensure_ascii=False),
                "question_type": item.metadata["question_type"],
                "hints_json": json.dumps(item.metadata["hints"], sort_keys=True),
                "tags_json": json.dumps(item.metadata["tags"], sort_keys=True),
                "raw_answer": raw_answer.text,
                "compressed_answer": compressed_answer.text,
                "raw_latency_ms": raw_answer.metadata["measured_latency_ms"],
                "compressed_latency_ms": compressed_answer.metadata["measured_latency_ms"],
                "compressor_latency_ms": item.compressed.metadata["measured_latency_ms"],
                "raw_tokens": raw_answer.tokens,
                "compressed_tokens": compressed_answer.tokens,
                "compressor_tokens": item.compressed.tokens,
                "raw_cost_usd": raw_answer.cost_usd,
                "compressed_cost_usd": compressed_answer.cost_usd,
                "compressor_cost_usd": item.compressed.cost_usd,
                "raw_adapter_metadata_json": json.dumps(json_safe(raw_answer.metadata), sort_keys=True),
                "compressed_adapter_metadata_json": json.dumps(
                    json_safe(compressed_answer.metadata), sort_keys=True
                ),
                "compressor_metadata_json": json.dumps(json_safe(item.compressed.metadata), sort_keys=True),
                "artifact_reused": item.artifact_reused,
                "raw_answer_reused": raw_reused,
                "compressed_answer_reused": compressed_reused,
                "judge_reused": raw_judged and compressed_judged,
            }
            for field_name in config.hint_fields:
                row[field_name] = hint_value(case, field_name)
            rows.append(row)
        return rows

    tasks = [(item, reader_key) for item in compiled for reader_key in config.readers]
    records = [row for rows in run_parallel(replay, tasks, config.workers) for row in rows]
    return ReplayResult(
        frame=pd.DataFrame(records),
        judge_calls=judge_usage["calls"],
        judge_cost_usd=judge_usage["cost_usd"],
    )
