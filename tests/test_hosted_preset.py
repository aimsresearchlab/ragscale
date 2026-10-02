import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ragscale.openai_compatible import parse_judge_verdict
from ragscale.runner import DEMO_DATASET, bundled_data_path, run_config, write_starter_project


def _fake_client_factory(calls, monkeypatch, *, judge_reply="1"):
    class FakeCompletions:
        def create(self, **request):
            calls.append(request)
            model = request["model"]
            if "compressor" in model:
                content = "gold evidence"
            elif "judge" in model:
                content = judge_reply
            else:
                content = "gold"
            return SimpleNamespace(
                id=f"response-{len(calls)}",
                model=model,
                provider="fake-provider",
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2, total_tokens=9, cost=0.001),
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("ragscale.openai_compatible._new_client", lambda **_: FakeClient())


def _hosted_project(tmp_path, *, workers=1, judge=False, rows=12):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = [
        {
            "example_id": f"q{index}",
            "question": "Return the gold token",
            "reference_answers": ["gold"],
            "candidate_pool": [{"text": "gold evidence"}, {"text": "noise"}],
            "question_type": "factoid",
        }
        for index in range(rows)
    ]
    (tmp_path / "eval.jsonl").write_text("\n".join(json.dumps(row) for row in dataset) + "\n")
    common = {"type": "openai_compatible", "base_url": "https://gateway.example/v1",
              "api_key_env": "RAGSCALE_TEST_API_KEY", "parameters": {"temperature": 0}}
    metrics = ["exact_match"]
    if judge:
        metrics.append({"name": "judge_match", "model": "test/judge", **common})
    config = {
        "version": 1,
        "dataset": {"path": "eval.jsonl", "name": "hosted-test", "minimum_slice_rows": 1},
        "pipeline": {"raw_policy_id": "raw-v1",
                     "compressor": {"id": "c", "model": "test/compressor", **common}},
        "readers": {
            "current": {"name": "a", "family": "fa", "model": "test/current", **common},
            "candidate": {"name": "b", "family": "fb", "model": "test/candidate", **common},
        },
        "comparison": {"current": "current", "candidate": "candidate"},
        "scoring": {"metrics": metrics},
        "run": {"output_dir": "audit", "bootstrap_draws": 20, "seed": 3, "workers": workers},
    }
    path = tmp_path / "ragscale.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_bundled_demo_dataset_is_valid():
    path = bundled_data_path()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 100
    assert len({row["example_id"] for row in rows}) == 100
    for row in rows:
        assert row["reference_answers"] and row["question"]
        assert 2 <= len(row["candidate_pool"]) <= 10


@pytest.mark.parametrize("preset", ["openrouter", "openai"])
def test_hosted_preset_writes_key_only_project(tmp_path, preset):
    config_path = write_starter_project(tmp_path / preset, preset=preset)
    config = yaml.safe_load(config_path.read_text())
    assert (tmp_path / preset / DEMO_DATASET).exists()
    assert config["pipeline"]["compressor"]["type"] == "openai_compatible"
    for reader in config["readers"].values():
        assert reader["type"] == "openai_compatible"
        assert reader["parameters"]["temperature"] == 0
    assert config["run"]["fail_on_outcomes"] == []
    assert "Illustration only" in config["dataset"]["description"]
    assert not any(path.suffix == ".py" for path in (tmp_path / preset).iterdir())


def test_unknown_preset_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown preset"):
        write_starter_project(tmp_path, preset="nope")


def test_call_cache_resumes_without_repeat_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGSCALE_TEST_API_KEY", "k")
    calls = []
    _fake_client_factory(calls, monkeypatch)
    config_path = _hosted_project(tmp_path)
    first = run_config(config_path)
    assert len(calls) == 12 + 12 * 2 * 2
    assert first["resource_summary"]["reader_calls_this_run"] == 48
    assert abs(first["resource_summary"]["reported_cost_usd"] - 0.001 * (12 + 48)) < 1e-9
    assert (tmp_path / "audit" / "call_cache.jsonl").exists()

    calls.clear()
    second = run_config(config_path)
    assert calls == []
    assert second["resource_summary"]["reader_calls_this_run"] == 0
    assert second["resource_summary"]["compressor_calls_this_run"] == 0
    assert second["resource_summary"]["reported_cost_usd"] == 0.0
    assert second["metrics"][0]["raw_upgrade"] == first["metrics"][0]["raw_upgrade"]
    assert second["metrics"][0]["current_raw"] == first["metrics"][0]["current_raw"] == 1.0
    record = json.loads((tmp_path / "audit" / "records.jsonl").read_text().splitlines()[0])
    assert record["raw_answer_reused"] and record["compressed_answer_reused"]
    assert json.loads(record["raw_adapter_metadata_json"])["provider"] == "fake-provider"


def test_workers_do_not_change_results(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGSCALE_TEST_API_KEY", "k")
    calls = []
    _fake_client_factory(calls, monkeypatch)
    serial = run_config(_hosted_project(tmp_path / "serial", workers=1))
    parallel = run_config(_hosted_project(tmp_path / "parallel", workers=4))
    assert serial["resource_summary"]["workers"] == 1
    assert parallel["resource_summary"]["workers"] == 4
    keep = {"raw_upgrade", "compressed_upgrade", "upgrade_retention", "rows"}
    assert {k: serial["metrics"][0][k] for k in keep} == {k: parallel["metrics"][0][k] for k in keep}
    serial_rows = Path(serial["output_dir"], "paired_scores.csv").read_text().splitlines()
    parallel_rows = Path(parallel["output_dir"], "paired_scores.csv").read_text().splitlines()
    assert [row.split(",")[:6] for row in serial_rows] == [row.split(",")[:6] for row in parallel_rows]


def test_openai_compatible_judge_scores_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGSCALE_TEST_API_KEY", "k")
    calls = []
    _fake_client_factory(calls, monkeypatch)
    summary = run_config(_hosted_project(tmp_path, judge=True, rows=4))
    judge_calls = [call for call in calls if call["model"] == "test/judge"]
    assert len(judge_calls) == 4 * 2 * 2
    assert "Predicted answer" in judge_calls[0]["messages"][1]["content"]
    judged = [m for m in summary["metrics"] if m["metric"] == "judge_match"][0]
    assert judged["current_raw"] == 1.0
    assert summary["provenance"]["adapter_provenance"]["judges"]["judge_match"]["role"] == "judge"
    assert summary["resource_summary"]["judge_calls_this_run"] == 16
    assert abs(summary["resource_summary"]["judge_cost_usd"] - 0.016) < 1e-9

    calls.clear()
    run_config(tmp_path / "ragscale.yaml")
    assert calls == []


def test_judge_verdict_parsing():
    assert parse_judge_verdict(" 1 ") == 1.0
    assert parse_judge_verdict("No.") == 0.0
    with pytest.raises(ValueError, match="not a 0/1 verdict"):
        parse_judge_verdict("The answer is partially right")


def test_rate_limit_is_retried_with_backoff(monkeypatch):
    from ragscale.openai_compatible import OpenAICompatibleAdapter

    monkeypatch.setenv("RAGSCALE_TEST_API_KEY", "k")
    monkeypatch.setattr("ragscale.openai_compatible._BACKOFF_SCALE", 0.0)
    attempts = []

    class Limited(Exception):
        status_code = 429

    class FakeCompletions:
        def create(self, **request):
            attempts.append(request)
            if len(attempts) < 3:
                raise Limited("Error code: 429 - rate-limited upstream")
            return SimpleNamespace(
                id="r", model=request["model"], provider="p",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("ragscale.openai_compatible._new_client", lambda **_: FakeClient())
    adapter = OpenAICompatibleAdapter(role="reader", model="m", api_key_env="RAGSCALE_TEST_API_KEY")
    assert adapter("q", "e")["text"] == "ok"
    assert len(attempts) == 3

    attempts.clear()
    strict = OpenAICompatibleAdapter(
        role="reader", model="m", api_key_env="RAGSCALE_TEST_API_KEY", rate_limit_retries=0
    )
    with pytest.raises(RuntimeError, match="429"):
        strict("q", "e")


def test_compressor_calls_resume_before_manifest_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGSCALE_TEST_API_KEY", "k")
    calls = []
    _fake_client_factory(calls, monkeypatch)
    config_path = _hosted_project(tmp_path, rows=6)
    run_config(config_path)
    compressor_calls = [c for c in calls if c["model"] == "test/compressor"]
    assert len(compressor_calls) == 6

    # Simulate an interrupted run: the call cache survives, the manifest does not.
    (tmp_path / "audit" / "artifact_manifest.jsonl").unlink()
    calls.clear()
    summary = run_config(config_path)
    assert [c for c in calls if c["model"] == "test/compressor"] == []
    assert summary["resource_summary"]["compressor_calls_this_run"] == 0
    assert summary["provenance"]["artifacts_reused"] == 6


def test_empty_reply_is_retried_then_handled_by_role(monkeypatch):
    from ragscale.openai_compatible import OpenAICompatibleAdapter

    monkeypatch.setenv("RAGSCALE_TEST_API_KEY", "k")
    replies = []

    class FakeCompletions:
        def create(self, **request):
            replies.append(request["model"])
            return SimpleNamespace(
                id="r", model=request["model"], provider="p",
                choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("ragscale.openai_compatible._new_client", lambda **_: FakeClient())
    reader = OpenAICompatibleAdapter(role="reader", model="m", api_key_env="RAGSCALE_TEST_API_KEY")
    result = reader("q", "e")
    assert result["text"] == ""
    assert result["metadata"]["empty_response"] is True
    assert result["metadata"]["empty_retries_used"] == 2
    assert len(replies) == 3

    replies.clear()
    compressor = OpenAICompatibleAdapter(role="compressor", model="m", api_key_env="RAGSCALE_TEST_API_KEY")
    with pytest.raises(ValueError, match="no visible text"):
        compressor("q", "e")
    assert len(replies) == 3
