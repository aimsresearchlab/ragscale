import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ragscale.runner import run_config, write_starter_project


ADAPTERS = '''
def compress(question, documents, metadata):
    return {"text": documents[0]["text"], "tokens": 4, "cost_usd": 0.01}

def current_reader(question, evidence, metadata, condition):
    if metadata["example_id"] in {"q1", "q3"} and "gold" in evidence:
        return {"answer": "gold", "tokens": 2, "cost_usd": 0.001}
    return "wrong"

def candidate_reader(question, evidence, metadata, condition):
    if "gold" in evidence:
        return {"answer": "gold", "tokens": 2, "cost_usd": 0.002}
    return "wrong"
'''


def _project(tmp_path: Path) -> Path:
    (tmp_path / "adapters.py").write_text(ADAPTERS)
    rows = []
    for index in range(1, 21):
        rows.append({
            "example_id": f"q{index}",
            "question": "Return the gold token",
            "reference_answers": ["gold"],
            "candidate_pool": [
                {"text": "gold evidence" if index <= 10 else "irrelevant"},
                {"text": "gold evidence"},
            ],
            "question_type": "factoid" if index <= 10 else "multi-hop",
            "domain": "support",
            "difficulty": "easy" if index <= 10 else "hard",
            "hints": {"tenant": "demo"},
        })
    with (tmp_path / "eval.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    config = {
        "version": 1,
        "dataset": {
            "path": "eval.jsonl", "name": "test-set",
            "hint_fields": ["question_type", "domain", "difficulty"],
            "minimum_slice_rows": 5,
            "hints": {"source": "unit-test"},
        },
        "pipeline": {
            "raw_policy_id": "raw-v1",
            "compressor": {"id": "compressor-v1", "callable": "adapters:compress"},
        },
        "readers": {
            "current": {"name": "reader-a", "family": "family-a", "callable": "adapters:current_reader"},
            "candidate": {"name": "reader-b", "family": "family-b", "callable": "adapters:candidate_reader"},
        },
        "comparison": {"current": "current", "candidate": "candidate"},
        "scoring": {"metrics": ["exact_match", "token_f1"]},
        "run": {"output_dir": "audit", "bootstrap_draws": 50, "seed": 7},
    }
    (tmp_path / "ragscale.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return tmp_path / "ragscale.yaml"


def test_yaml_runner_compiles_once_scores_and_slices(tmp_path):
    summary = run_config(_project(tmp_path))
    assert summary["dataset"]["name"] == "test-set"
    assert summary["provenance"]["compressed_artifacts"] == 20
    assert summary["provenance"]["unique_compressed_artifact_hashes"] == 2
    assert summary["resource_summary"]["reported_cost_usd"] == pytest.approx(0.264)
    assert {result["metric"] for result in summary["metrics"]} == {"exact_match", "token_f1"}
    assert all(result["slices"] for result in summary["metrics"])
    output = Path(summary["output_dir"])
    assert (output / "summary.json").exists()
    assert (output / "REPORT.md").exists()
    assert (output / "report.html").exists()
    assert (output / "paired_scores.csv").exists()
    manifest = (output / "artifact_manifest.jsonl").read_text().splitlines()
    assert len(manifest) == 20
    assert all("sha256:" in line for line in manifest)
    replay = run_config(tmp_path / "ragscale.yaml")
    assert replay["provenance"]["artifacts_reused"] == 20
    assert replay["resource_summary"]["compressor_calls_this_run"] == 0


def test_starter_project_is_runnable(tmp_path):
    config = write_starter_project(tmp_path / "starter")
    summary = run_config(config)
    assert summary["dataset"]["rows"] == 2
    assert Path(summary["output_dir"], "summary.json").exists()


def test_duplicate_example_ids_are_rejected(tmp_path):
    config = _project(tmp_path)
    line = (tmp_path / "eval.jsonl").read_text().splitlines()[0]
    (tmp_path / "eval.jsonl").write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="must be unique"):
        run_config(config)


def test_openai_compatible_yaml_adapters_use_configured_base_url(
    tmp_path, monkeypatch
):
    config_path = _project(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    endpoint = "https://gateway.example/v1"
    common = {
        "type": "openai_compatible",
        "base_url": endpoint,
        "api_key_env": "RAGSCALE_TEST_API_KEY",
        "parameters": {"temperature": 0},
    }
    config["pipeline"]["compressor"] = {
        **common, "id": "remote-compressor", "model": "test/compressor"
    }
    config["readers"]["current"] = {
        **common, "name": "reader-a", "family": "family-a", "model": "test/current"
    }
    config["readers"]["candidate"] = {
        **common, "name": "reader-b", "family": "family-b", "model": "test/candidate"
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setenv("RAGSCALE_TEST_API_KEY", "test-key")

    client_settings = []
    requests = []

    class FakeCompletions:
        def create(self, **request):
            requests.append(request)
            is_compressor = request["model"] == "test/compressor"
            content = "gold evidence" if is_compressor else "gold"
            return SimpleNamespace(
                id=f"response-{len(requests)}",
                model=request["model"],
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(
                    prompt_tokens=7, completion_tokens=2, total_tokens=9
                ),
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    def fake_client(**settings):
        client_settings.append(settings)
        return FakeClient()

    monkeypatch.setattr("ragscale.openai_compatible._new_client", fake_client)
    summary = run_config(config_path)

    assert len(client_settings) == 3
    assert {settings["base_url"] for settings in client_settings} == {endpoint}
    assert all(settings["api_key"] == "test-key" for settings in client_settings)
    assert len(requests) == 100
    assert all(request["stream"] is False for request in requests)
    provenance = summary["provenance"]["adapter_provenance"]
    assert provenance["compressor"]["model"] == "test/compressor"
    assert provenance["readers"]["candidate"]["base_url"] == endpoint
    records = Path(summary["output_dir"], "records.jsonl").read_text().splitlines()
    first = json.loads(records[0])
    raw_metadata = json.loads(first["raw_adapter_metadata_json"])
    assert raw_metadata["base_url"] == endpoint
    assert raw_metadata["response_id"].startswith("response-")


def test_openai_compatible_adapter_requires_named_api_key(tmp_path, monkeypatch):
    config_path = _project(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["pipeline"]["compressor"] = {
        "id": "remote-compressor",
        "type": "openai_compatible",
        "model": "test/compressor",
        "api_key_env": "RAGSCALE_MISSING_TEST_KEY",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.delenv("RAGSCALE_MISSING_TEST_KEY", raising=False)

    with pytest.raises(ValueError, match="RAGSCALE_MISSING_TEST_KEY"):
        run_config(config_path)


def test_two_projects_with_same_adapter_module_name_do_not_collide(tmp_path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first = run_config(_project(tmp_path / "first"))
    _project(tmp_path / "second")
    (tmp_path / "second" / "adapters.py").write_text(ADAPTERS.replace('"gold"', '"wrong"'))
    second = run_config(tmp_path / "second" / "ragscale.yaml")
    assert first["metrics"][0]["candidate_raw"] == 1.0
    assert second["metrics"][0]["candidate_raw"] == 0.0
