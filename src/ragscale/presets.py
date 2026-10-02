"""Starter projects: a local stub project, or a key-only hosted preset."""

from __future__ import annotations

from pathlib import Path
from string import Template

_INSTALLED_DATA_DIR = Path(__file__).resolve().parent / "data"
_SOURCE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEMO_DATASET = "hotpotqa_demo_100.jsonl"


def bundled_data_path(name: str = DEMO_DATASET) -> Path:
    """Locate a bundled data file in an installed or source checkout."""
    for directory in (_INSTALLED_DATA_DIR, _SOURCE_DATA_DIR):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Bundled data file is missing: {name}")


# Hosted presets. Each adapter reads its key from api_key_env and sends
# temperature 0. OpenRouter additionally reports per-call cost and, for the
# Qwen3 compressor, needs hidden reasoning switched off in two places.
PRESETS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "compressor": "qwen/qwen3-14b",
        "current": ("qwen/qwen-2.5-7b-instruct", "qwen"),
        "candidate": ("meta-llama/llama-3.3-70b-instruct", "meta-llama"),
        "judge": "openai/gpt-4.1-mini",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "compressor": "gpt-4.1-mini",
        "current": ("gpt-4.1-nano", "openai"),
        "candidate": ("gpt-4.1-mini", "openai"),
        "judge": "gpt-4.1-mini",
    },
}

_OPENROUTER_USAGE = """
      extra_body:
        usage:
          include: true"""

_OPENROUTER_COMPRESSOR = """
      # OpenRouter-specific: keep hidden reasoning out of the summary budget.
      extra_body:
        usage:
          include: true
        reasoning:
          enabled: false
    # Qwen3 prompt switch; some providers ignore the reasoning flag above.
    user_prompt_template: |-
      Question:
      {question}

      Evidence:
      {evidence}

      /no_think"""

_OPENROUTER_JUDGE = """
    #     extra_body:
    #       usage:
    #         include: true"""

HOSTED_TEMPLATE = Template("""version: 1

# Key-only demonstration. Set $api_key_env and run: ragscale run ragscale.yaml
# Every compressor and reader call goes to $base_url.
# This run is an illustration on one fresh summary draw and two readers. It is
# not evidence about any other reader, compressor, or benchmark. Replace the
# compressor and readers with your deployed pipeline for a real audit.

dataset:
  path: $dataset
  name: hotpotqa-demo-100
  description: >-
    Illustration only. First 100 rows of the paper's frozen HotpotQA distractor
    slice, one fresh LLM-summary artifact per row, two hosted readers. Not
    paper evidence and not a verdict about other readers or compressors.
  hint_fields: [question_type, difficulty]
  minimum_slice_rows: 10
  hints:
    source: hotpotqa_distractor_500_v1
    language: en

pipeline:
  raw_policy_id: hotpotqa-distractor-10
  compressor:
    id: llm-summary:$compressor
    type: openai_compatible
    model: $compressor
    base_url: $base_url
    api_key_env: $api_key_env
    parameters:
      temperature: 0
      max_tokens: 1024$compressor_extra

readers:
  current:
    name: $current_model
    family: $current_family
    type: openai_compatible
    model: $current_model
    base_url: $base_url
    api_key_env: $api_key_env
    parameters:
      temperature: 0
      max_tokens: 64$reader_extra
  candidate:
    name: $candidate_model
    family: $candidate_family
    type: openai_compatible
    model: $candidate_model
    base_url: $base_url
    api_key_env: $api_key_env
    parameters:
      temperature: 0
      max_tokens: 64$reader_extra

comparison:
  current: current
  candidate: candidate

scoring:
  metrics:
    - exact_match
    - token_f1
    # Optional model judge for free-form answers. It adds one call per answer.
    # - name: judge_match
    #   type: openai_compatible
    #   model: $judge
    #   base_url: $base_url
    #   api_key_env: $api_key_env
    #   parameters:
    #     temperature: 0
    #     max_tokens: 4$judge_extra

run:
  output_dir: audit
  bootstrap_draws: 2000
  seed: 20260821
  minimum_raw_gap: 0.05
  retention_alert_threshold: 0.75
  workers: 8
  store_artifact_text: true
  reuse_artifacts: true
  reuse_answers: true
  fail_on_outcomes: []
""")


def hosted_preset_config(preset: str) -> str:
    settings = PRESETS[preset]
    hosted = preset == "openrouter"
    return HOSTED_TEMPLATE.substitute(
        dataset=DEMO_DATASET,
        base_url=settings["base_url"],
        api_key_env=settings["api_key_env"],
        compressor=settings["compressor"],
        current_model=settings["current"][0],
        current_family=settings["current"][1],
        candidate_model=settings["candidate"][0],
        candidate_family=settings["candidate"][1],
        judge=settings["judge"],
        compressor_extra=_OPENROUTER_COMPRESSOR if hosted else "",
        reader_extra=_OPENROUTER_USAGE if hosted else "",
        judge_extra=_OPENROUTER_JUDGE if hosted else "",
    )


LOCAL_CONFIG = """version: 1

dataset:
  path: eval.jsonl
  name: support-qa
  description: Small example; replace with a held-out or sampled production evaluation set.
  hint_fields: [question_type, domain, difficulty]
  minimum_slice_rows: 1
  hints:
    source: local-starter
    language: en

pipeline:
  raw_policy_id: retrieved-candidate-pool-v1
  compressor:
    id: deployed-compressor-v1
    callable: adapters:compress

readers:
  current:
    name: current-reader
    family: current-family
    callable: adapters:current_reader
  candidate:
    name: candidate-reader
    family: candidate-family
    callable: adapters:candidate_reader

comparison:
  current: current
  candidate: candidate

scoring:
  metrics: [exact_match, token_f1, token_precision, token_recall]

run:
  output_dir: audit
  bootstrap_draws: 1000
  seed: 20260821
  minimum_raw_gap: 0.05
  retention_alert_threshold: 0.75
  store_artifact_text: true
  reuse_artifacts: true
  fail_on_outcomes: []
"""

LOCAL_DATASET = """{"example_id":"q1","question":"What city is the Eiffel Tower in?","reference_answers":["Paris"],"candidate_pool":[{"text":"The Eiffel Tower is in Paris, France."},{"text":"Berlin is the capital of Germany."}],"question_type":"factoid","domain":"travel","difficulty":"easy","tags":["geography"]}
{"example_id":"q2","question":"Who wrote Pride and Prejudice?","reference_answers":["Jane Austen"],"candidate_pool":[{"text":"Charles Dickens wrote Great Expectations."},{"text":"Pride and Prejudice was written by Jane Austen."}],"question_type":"factoid","domain":"literature","difficulty":"easy","tags":["books"]}
"""

LOCAL_ADAPTERS = '''"""Replace these functions with adapters around the deployed compressor and readers."""

def compress(question, documents, metadata):
    texts = [document.get("text", str(document)) if isinstance(document, dict) else str(document) for document in documents]
    return {"text": "\\n".join(texts[:1]), "tokens": len(texts[0].split())}

def _answer(question, evidence):
    if "Eiffel" in question and "Paris" in evidence:
        return "Paris"
    if "Pride and Prejudice" in question and "Jane Austen" in evidence:
        return "Jane Austen"
    return "unknown"

def current_reader(question, evidence, metadata, condition):
    if "Eiffel" in question and "Paris" in evidence:
        return {"answer": "Paris"}
    return {"answer": "unknown"}

def candidate_reader(question, evidence, metadata, condition):
    return {"answer": _answer(question, evidence)}
'''


def write_starter_project(output_dir: str | Path, preset: str = "local") -> Path:
    """Create a runnable starter: local stub adapters, or a hosted key-only preset."""
    if preset not in {"local", *PRESETS}:
        raise ValueError(f"Unknown preset {preset!r}; choose local, openrouter, or openai")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if preset == "local":
        files = {"ragscale.yaml": LOCAL_CONFIG, "eval.jsonl": LOCAL_DATASET, "adapters.py": LOCAL_ADAPTERS}
    else:
        files = {
            "ragscale.yaml": hosted_preset_config(preset),
            DEMO_DATASET: bundled_data_path().read_text(encoding="utf-8"),
        }
    for name, contents in files.items():
        path = output / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing starter file: {path}")
        path.write_text(contents, encoding="utf-8")
    return output / "ragscale.yaml"
