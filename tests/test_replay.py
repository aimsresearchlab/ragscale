from pathlib import Path

import pandas as pd
import pytest

from ragscale.replay import replay_audit


def panel(readers=8):
    rows = []
    for r in range(readers):
        for i in range(40):
            raw = int(i < 8 + 3 * r)
            compressed = int(i < 20 + r)
            rows.append({
                "example_id": f"q{i}", "reader": f"r{r}",
                "reader_family": f"f{r % 4}", "raw_score": raw,
                "compressed_score": compressed, "candidate_pool_id": f"pool-{i}",
                "compressed_artifact_hash": f"artifact-{i}",
                "raw_policy_id": "raw-v1", "metric": "exact_match",
            })
    return pd.DataFrame(rows)


def test_replay_audit_reports_broader_panel():
    result = replay_audit(panel(), splits=20)
    assert result.outcome == "observed-panel-audited"
    assert result.readers == 8
    assert result.families == 4
    assert result.endpoint_retention < 1


def test_three_readers_are_only_an_illustration():
    result = replay_audit(panel(3), splits=5)
    assert result.outcome == "observed-panel-audited"
    assert result.warnings


def test_two_readers_are_an_observed_pair():
    result = replay_audit(panel(2), splits=5)
    assert result.outcome == "observed-pair-audited"
    assert "does not screen" in result.warnings[0]


def test_rejects_reader_specific_artifact():
    frame = panel()
    frame.loc[(frame.example_id == "q0") & (frame.reader == "r0"), "compressed_artifact_hash"] = "different"
    with pytest.raises(ValueError, match="differs across readers"):
        replay_audit(frame, splits=5)


def test_rejects_incomplete_footprint():
    frame = panel().iloc[1:].copy()
    with pytest.raises(ValueError, match="exact item footprint"):
        replay_audit(frame, splits=5)


def test_multi_metric_input_requires_metric_choice():
    frame = pd.concat([panel(), panel().assign(metric="token_f1")], ignore_index=True)
    with pytest.raises(ValueError, match="several metrics"):
        replay_audit(frame, splits=5)
    with pytest.raises(ValueError, match="not in the input"):
        replay_audit(frame, splits=5, metric="rouge_l")
    result = replay_audit(frame, splits=5, metric="token_f1")
    assert result.metric == "token_f1"
    assert result.readers == 8


def test_run_output_feeds_replay_audit(tmp_path):
    from ragscale.runner import run_config, write_starter_project

    summary = run_config(write_starter_project(tmp_path / "starter"))
    scores = Path(summary["output_dir"]) / "paired_scores.csv"
    result = replay_audit(scores, splits=5, metric="exact_match")
    assert result.outcome == "observed-pair-audited"
    assert result.rows == summary["dataset"]["rows"]
