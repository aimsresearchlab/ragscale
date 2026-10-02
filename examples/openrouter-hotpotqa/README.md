# Key-only OpenRouter example

This directory is the output of the hosted preset, run once on 2026-09-01:

```bash
export OPENROUTER_API_KEY=...
ragscale init --preset openrouter --output examples/openrouter-hotpotqa
ragscale run examples/openrouter-hotpotqa/ragscale.yaml
```

The only change from the generated `ragscale.yaml` is that the optional
model judge (`openai/gpt-4.1-mini`) is enabled. Everything under `audit/` was
written by `ragscale run`. Nothing was edited by hand.

## What it is

An illustration that installation to report works on a network pipeline
with no Python adapters. It is not evidence about any reader, compressor, or
benchmark, and it is not cited as a number in the paper.

| Component | Setting |
| --- | --- |
| Rows | first 100 of the frozen HotpotQA distractor slice (`hotpotqa_demo_100.jsonl`) |
| Compressor | `qwen/qwen3-14b`, one LLM summary per row, reasoning off, temperature 0 |
| Current reader | `qwen/qwen-2.5-7b-instruct`, temperature 0 |
| Candidate reader | `meta-llama/llama-3.3-70b-instruct`, temperature 0 |
| Metrics | exact match, token F1, model judge |
| Calls | 100 compressor, 400 reader, 400 judge, 8 workers |
| Wall time | 2 minutes 20 seconds |
| Reported cost | $0.07 compressor and readers, $0.02 judge |

## What it shows

| Metric | Current compression gain | Raw upgrade | Compressed upgrade | Retention |
| --- | ---: | ---: | ---: | ---: |
| Exact match | +11.0pp | +9.0pp | +3.0pp | 0.33 |
| Token F1 | +11.3pp | +9.5pp | +4.4pp | 0.47 |
| Judge | +13.0pp | +12.0pp | +4.0pp | 0.33 |

The summary helps the current reader on every metric, and the visible
current-to-candidate upgrade shrinks under it. That is the shape the paper
describes. The retention intervals over 2,000 row bootstraps all cross 1,
because 100 rows cannot bound a ratio of two small differences. Read the
outcome labels as descriptions of this run, not as a test.

Judge and exact match agree on the direction. Exact match scores the current
reader at 53% raw and the judge at 76%, so a third of its raw errors under
exact match are wording rather than content. Compare both rows before reading
rescue and damage.

## Things to know before reusing it

- OpenRouter routes each call to one of several providers. Two runs at
  temperature 0 differed by one or two rows per reader. Pin a provider with
  `parameters.extra_body.provider` if you need repeatable answers.
- The first run of this preset returned empty summaries from providers that
  ignore the reasoning-off flag. The preset now appends `/no_think` to the
  compressor prompt. If you change the compressor model, check
  `artifact_manifest.jsonl` for empty `compressed_evidence`.
- `call_cache.jsonl` holds every reader and judge reply. Rerunning in this
  directory makes no network calls and reports zero cost. Delete `audit/` for
  a fresh draw.
- `records.jsonl` and `paired_scores.csv` contain the HotpotQA text and every
  answer. They are committed so the run can be inspected without a key.
