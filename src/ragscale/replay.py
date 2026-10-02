"""Fixed-artifact replay audit for paired reader outcomes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "example_id",
    "reader",
    "reader_family",
    "raw_score",
    "compressed_score",
    "candidate_pool_id",
    "compressed_artifact_hash",
    "raw_policy_id",
    "metric",
}


@dataclass
class ReplayAuditResult:
    readers: int
    families: int
    rows: int
    metric: str
    raw_policy_id: str
    endpoint_readers: list[str]
    raw_endpoint_span: float
    compressed_endpoint_span: float
    endpoint_retention: float
    full_sample_eligible_pairs: int
    full_sample_compressed_reversals: int
    median_raw_replay_flips: float
    median_compressed_flips: float
    median_excess_flips: float
    excess_flip_split_interval: list[float]
    rescue_share: float | None
    raw_correct_damage: float | None
    outcome: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Fixed-Artifact Reader Replay", "",
            f"**Outcome:** {self.outcome}", "",
            f"- Readers: {self.readers} from {self.families} families",
            f"- Shared rows: {self.rows}",
            f"- Metric: {self.metric}",
            f"- Raw policy: `{self.raw_policy_id}`",
            f"- Raw-selected endpoints: {self.endpoint_readers[0]} to {self.endpoint_readers[1]}",
            f"- Raw endpoint span: {100 * self.raw_endpoint_span:.1f}pp",
            f"- Compressed endpoint span: {100 * self.compressed_endpoint_span:.1f}pp",
            f"- Endpoint retention: {100 * self.endpoint_retention:.1f}%",
            f"- Median excess flips: {100 * self.median_excess_flips:.1f}pp",
            "",
            "This outcome describes only the supplied readers and rows. No reader-count threshold certifies unseen readers or broader ranking stability.",
        ]
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines) + "\n"


def _require_fixed_artifacts(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Input contains no rows")
    if frame.duplicated(["example_id", "reader"]).any():
        raise ValueError("Duplicate example_id and reader keys")
    for column in ("raw_score", "compressed_score"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must contain finite scores in [0, 1]")
        frame[column] = values
    for column in ("candidate_pool_id", "compressed_artifact_hash"):
        counts = frame.groupby("example_id")[column].nunique(dropna=False)
        if (counts != 1).any():
            bad = str(counts[counts != 1].index[0])
            raise ValueError(f"{column} differs across readers for item {bad}")
    for column in ("raw_policy_id", "metric"):
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"Input must contain exactly one {column}")
    family_counts = frame.groupby("reader")["reader_family"].nunique(dropna=False)
    if (family_counts != 1).any():
        raise ValueError("A reader maps to multiple reader families")
    expected = frame["example_id"].nunique()
    coverage = frame.groupby("reader")["example_id"].nunique()
    if (coverage != expected).any():
        raise ValueError("Readers do not share an exact item footprint")


def _flip_rates(raw: np.ndarray, comp: np.ndarray, selection: np.ndarray, scoring: np.ndarray, gap: float) -> tuple[float, float, int]:
    raw_select = raw[:, selection].mean(axis=1)
    raw_score = raw[:, scoring].mean(axis=1)
    comp_score = comp[:, scoring].mean(axis=1)
    eligible = raw_flips = comp_flips = 0
    for i, j in combinations(range(len(raw)), 2):
        direction = np.sign(raw_select[j] - raw_select[i])
        if abs(raw_select[j] - raw_select[i]) < gap:
            continue
        eligible += 1
        raw_flips += int(direction * (raw_score[j] - raw_score[i]) < 0)
        comp_flips += int(direction * (comp_score[j] - comp_score[i]) < 0)
    if not eligible:
        return math.nan, math.nan, 0
    return raw_flips / eligible, comp_flips / eligible, eligible


def _select_metric(frame: pd.DataFrame, metric: str | None) -> pd.DataFrame:
    """Keep one metric's rows; a multi-metric file must name which one."""
    if "metric" not in frame.columns:
        return frame
    available = sorted(frame["metric"].dropna().astype(str).unique())
    if metric is None:
        if len(available) > 1:
            raise ValueError(
                "Input contains several metrics; pass metric= (or --metric) to choose one of: "
                + ", ".join(available)
            )
        return frame
    if metric not in available:
        raise ValueError(f"Metric {metric!r} is not in the input; available: {', '.join(available)}")
    return frame[frame["metric"].astype(str) == metric].copy()


def replay_audit(
    data: str | Path | pd.DataFrame,
    *,
    splits: int = 5_000,
    seed: int = 20260821,
    minimum_raw_gap: float = 0.05,
    metric: str | None = None,
) -> ReplayAuditResult:
    """Audit one fixed compressed artifact across a shared reader panel.

    ``data`` may hold several metrics, as ``ragscale run`` writes to
    ``paired_scores.csv``; ``metric`` selects one of them.
    """
    frame = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data.copy()
    frame = _select_metric(frame, metric)
    _require_fixed_artifacts(frame)
    readers = sorted(frame["reader"].unique())
    if len(readers) < 2:
        raise ValueError("At least two readers are required")
    ids = sorted(frame["example_id"].astype(str).unique())
    indexed = frame.assign(example_id=frame.example_id.astype(str)).set_index(["reader", "example_id"])
    raw = np.asarray([[indexed.loc[(reader, item), "raw_score"] for item in ids] for reader in readers], dtype=float)
    comp = np.asarray([[indexed.loc[(reader, item), "compressed_score"] for item in ids] for reader in readers], dtype=float)
    raw_means, comp_means = raw.mean(axis=1), comp.mean(axis=1)
    weak = min(range(len(readers)), key=lambda i: (raw_means[i], readers[i]))
    strong = min(range(len(readers)), key=lambda i: (-raw_means[i], readers[i]))
    raw_span = float(raw_means[strong] - raw_means[weak])
    comp_span = float(comp_means[strong] - comp_means[weak])
    retention = comp_span / raw_span if raw_span >= minimum_raw_gap else math.nan

    all_idx = np.arange(len(ids))
    _, full_comp_rate, full_eligible = _flip_rates(raw, comp, all_idx, all_idx, minimum_raw_gap)
    full_reversals = int(round(full_comp_rate * full_eligible)) if full_eligible else 0
    rng = np.random.default_rng(seed)
    raw_rates, comp_rates, excess_rates = [], [], []
    for _ in range(splits):
        order = rng.permutation(len(ids))
        cut = len(ids) // 2
        raw_rate, comp_rate, eligible = _flip_rates(
            raw, comp, order[:cut], order[cut:], minimum_raw_gap
        )
        if not eligible:
            continue
        raw_rates.append(raw_rate)
        comp_rates.append(comp_rate)
        excess_rates.append(comp_rate - raw_rate)

    warnings = []
    families = frame[["reader", "reader_family"]].drop_duplicates()["reader_family"].nunique()
    if len(readers) == 2:
        warnings.append("This is an observed pair comparison. It does not screen a broader reader ranking.")
    else:
        warnings.append("No fixed reader count has been validated as sufficient to screen unseen readers or all broader-panel reversals.")
    if families < 3:
        warnings.append("Fewer than three reader families.")
    if not np.isfinite(retention):
        warnings.append("The raw endpoint gap is below 5pp, so endpoint retention is inadmissible.")
    binary = set(frame.raw_score.unique()) <= {0, 1} and set(frame.compressed_score.unique()) <= {0, 1}
    rescue = damage = None
    if binary:
        rescue = float(((frame.raw_score == 0) & (frame.compressed_score == 1)).mean())
        raw_correct = frame.raw_score == 1
        damage = float(((raw_correct) & (frame.compressed_score == 0)).sum() / raw_correct.sum()) if raw_correct.any() else math.nan
    outcome = "observed-pair-audited" if len(readers) == 2 else "observed-panel-audited"
    return ReplayAuditResult(
        readers=len(readers), families=int(families), rows=len(ids),
        metric=str(frame.metric.iloc[0]), raw_policy_id=str(frame.raw_policy_id.iloc[0]),
        endpoint_readers=[readers[weak], readers[strong]], raw_endpoint_span=raw_span,
        compressed_endpoint_span=comp_span, endpoint_retention=retention,
        full_sample_eligible_pairs=full_eligible,
        full_sample_compressed_reversals=full_reversals,
        median_raw_replay_flips=float(np.median(raw_rates)) if raw_rates else math.nan,
        median_compressed_flips=float(np.median(comp_rates)) if comp_rates else math.nan,
        median_excess_flips=float(np.median(excess_rates)) if excess_rates else math.nan,
        excess_flip_split_interval=[float(np.percentile(excess_rates, 2.5)), float(np.percentile(excess_rates, 97.5))] if excess_rates else [math.nan, math.nan],
        rescue_share=rescue, raw_correct_damage=damage, outcome=outcome, warnings=warnings,
    )


def write_replay_audit(result: ReplayAuditResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    (output / "REPORT.md").write_text(result.to_markdown())
