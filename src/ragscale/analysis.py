"""Pair statistics: upgrade retention, bootstrap intervals, transitions, and slices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ragscale.replay import replay_audit


@dataclass
class MetricComparison:
    metric: str
    current_reader: str
    candidate_reader: str
    rows: int
    current_raw: float
    current_compressed: float
    current_compression_gain: float
    candidate_raw: float
    candidate_compressed: float
    candidate_compression_gain: float
    raw_upgrade: float
    compressed_upgrade: float
    upgrade_retention: float
    raw_upgrade_interval: list[float]
    compressed_upgrade_interval: list[float]
    retention_interval: list[float]
    order_reversed: bool
    current_rescue: float | None
    current_damage: float | None
    candidate_rescue: float | None
    candidate_damage: float | None
    retention_alert_threshold: float
    outcome: str
    panel: dict[str, Any] | None
    slices: list[dict[str, Any]]
    changed_examples: list[dict[str, Any]]


def bootstrap_pair(
    current_raw: np.ndarray, current_comp: np.ndarray,
    candidate_raw: np.ndarray, candidate_comp: np.ndarray,
    *, draws: int, seed: int, minimum_gap: float,
) -> tuple[list[float], list[float], list[float]]:
    """Row bootstrap of the raw upgrade, compressed upgrade, and their ratio."""
    rng = np.random.default_rng(seed)
    raw_values, compressed_values, retention_values = [], [], []
    rows = len(current_raw)
    for _ in range(draws):
        indices = rng.integers(0, rows, size=rows)
        raw_upgrade = float(np.mean(candidate_raw[indices] - current_raw[indices]))
        compressed_upgrade = float(np.mean(candidate_comp[indices] - current_comp[indices]))
        raw_values.append(raw_upgrade)
        compressed_values.append(compressed_upgrade)
        if abs(raw_upgrade) >= minimum_gap:
            retention_values.append(compressed_upgrade / raw_upgrade)

    def interval(values: list[float]) -> list[float]:
        if not values:
            return [math.nan, math.nan]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return interval(raw_values), interval(compressed_values), interval(retention_values)


def transition_rates(raw: np.ndarray, compressed: np.ndarray) -> tuple[float | None, float | None]:
    """Rescue share and raw-correct damage; only defined for 0/1 scores."""
    if not set(np.unique(raw)).issubset({0.0, 1.0}) or not set(np.unique(compressed)).issubset({0.0, 1.0}):
        return None, None
    rescue = float(np.mean((raw == 0) & (compressed == 1)))
    correct = raw == 1
    damage = float(np.mean(compressed[correct] == 0)) if np.any(correct) else math.nan
    return rescue, damage


def slice_rows(
    metric_frame: pd.DataFrame, current: str, candidate: str, hint_fields: list[str],
    minimum_rows: int, minimum_gap: float,
) -> list[dict[str, Any]]:
    """Raw and compressed upgrade within each hint value that has enough rows.

    ``metric_frame`` must hold one metric; retention is reported only when the
    slice's raw upgrade reaches ``minimum_gap``, as for the whole-panel figure.
    """
    metric = str(metric_frame.metric.iloc[0])
    results = []
    for field_name in hint_fields:
        if field_name not in metric_frame:
            continue
        for value, group in metric_frame.groupby(field_name, dropna=False):
            current_rows = group[group.reader == current]
            candidate_rows = group[group.reader == candidate]
            if len(current_rows) != len(candidate_rows) or group.example_id.nunique() < minimum_rows:
                continue
            current_rows = current_rows.sort_values("example_id")
            candidate_rows = candidate_rows.sort_values("example_id")
            raw_upgrade = float((candidate_rows.raw_score.to_numpy() - current_rows.raw_score.to_numpy()).mean())
            comp_upgrade = float((candidate_rows.compressed_score.to_numpy() - current_rows.compressed_score.to_numpy()).mean())
            results.append({
                "metric": metric,
                "field": field_name,
                "value": str(value),
                "rows": int(len(current_rows)),
                "raw_upgrade": raw_upgrade,
                "compressed_upgrade": comp_upgrade,
                "retention": comp_upgrade / raw_upgrade if abs(raw_upgrade) >= minimum_gap else math.nan,
            })
    return results


def _transition_label(raw: float, compressed: float) -> str:
    if raw == 0 and compressed == 1:
        return "rescued"
    if raw == 1 and compressed == 0:
        return "damaged"
    return "unchanged"


def _changed_examples(current_rows: pd.DataFrame, candidate_rows: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    joined = current_rows.merge(candidate_rows, on="example_id", suffixes=("_current", "_candidate"))
    changed = []
    for _, row in joined.iterrows():
        current = _transition_label(row.raw_score_current, row.compressed_score_current)
        candidate = _transition_label(row.raw_score_candidate, row.compressed_score_candidate)
        if current == "unchanged" and candidate == "unchanged":
            continue
        changed.append({
            "example_id": str(row.example_id),
            "question_type": str(row.get("question_type_current", "unspecified")),
            "current_transition": current,
            "candidate_transition": candidate,
            "current_raw_answer": str(row.raw_answer_current),
            "current_compressed_answer": str(row.compressed_answer_current),
            "candidate_raw_answer": str(row.raw_answer_candidate),
            "candidate_compressed_answer": str(row.compressed_answer_candidate),
        })
        if len(changed) == limit:
            break
    return changed


def analyze_metric(
    frame: pd.DataFrame, metric: str, current: str, candidate: str, *,
    bootstrap_draws: int, seed: int, minimum_gap: float, retention_alert_threshold: float,
    hint_fields: list[str], minimum_slice_rows: int,
) -> MetricComparison:
    """Compare the current and candidate readers on one metric."""
    metric_frame = frame[frame.metric == metric].copy()
    by_reader = {reader: rows.sort_values("example_id") for reader, rows in metric_frame.groupby("reader")}
    if current not in by_reader or candidate not in by_reader:
        raise ValueError(f"Metric {metric} lacks the configured current or candidate reader")
    current_rows, candidate_rows = by_reader[current], by_reader[candidate]
    if current_rows.example_id.tolist() != candidate_rows.example_id.tolist():
        raise ValueError(f"Metric {metric} has mismatched current/candidate footprints")
    cr = current_rows.raw_score.to_numpy(float)
    cc = current_rows.compressed_score.to_numpy(float)
    nr = candidate_rows.raw_score.to_numpy(float)
    nc = candidate_rows.compressed_score.to_numpy(float)
    current_raw, current_comp = float(cr.mean()), float(cc.mean())
    candidate_raw, candidate_comp = float(nr.mean()), float(nc.mean())
    raw_upgrade = candidate_raw - current_raw
    compressed_upgrade = candidate_comp - current_comp
    retention = compressed_upgrade / raw_upgrade if abs(raw_upgrade) >= minimum_gap else math.nan
    raw_interval, compressed_interval, retention_interval = bootstrap_pair(
        cr, cc, nr, nc, draws=bootstrap_draws, seed=seed, minimum_gap=minimum_gap
    )
    current_rescue, current_damage = transition_rates(cr, cc)
    candidate_rescue, candidate_damage = transition_rates(nr, nc)
    order_reversed = raw_upgrade * compressed_upgrade < 0
    if abs(raw_upgrade) < minimum_gap:
        outcome = "insufficient-raw-separation"
    elif order_reversed:
        outcome = "observed-order-reversed"
    elif retention < retention_alert_threshold:
        outcome = "observed-upgrade-attenuated"
    else:
        outcome = "observed-upgrade-preserved"

    panel = None
    if metric_frame.reader.nunique() > 2:
        replay = replay_audit(
            metric_frame, splits=min(bootstrap_draws, 5_000), seed=seed, minimum_raw_gap=minimum_gap
        )
        panel = replay.to_dict()
    slices = slice_rows(
        metric_frame, current, candidate, hint_fields, minimum_slice_rows, minimum_gap
    )
    changed = _changed_examples(current_rows, candidate_rows) if metric in {"exact_match", "em"} else []
    return MetricComparison(
        metric=metric, current_reader=current, candidate_reader=candidate,
        rows=len(current_rows), current_raw=current_raw, current_compressed=current_comp,
        current_compression_gain=current_comp - current_raw,
        candidate_raw=candidate_raw, candidate_compressed=candidate_comp,
        candidate_compression_gain=candidate_comp - candidate_raw,
        raw_upgrade=raw_upgrade, compressed_upgrade=compressed_upgrade,
        upgrade_retention=retention, raw_upgrade_interval=raw_interval,
        compressed_upgrade_interval=compressed_interval, retention_interval=retention_interval,
        order_reversed=order_reversed, current_rescue=current_rescue,
        current_damage=current_damage, candidate_rescue=candidate_rescue,
        candidate_damage=candidate_damage, outcome=outcome, panel=panel, slices=slices,
        retention_alert_threshold=retention_alert_threshold,
        changed_examples=changed,
    )
