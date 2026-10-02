# Changelog

## 0.5.1, 2026-09-05

- `replay-audit` accepts `--metric` (`replay_audit(metric=...)`). The `paired_scores.csv` written by `ragscale run` holds one row per metric, so the documented handoff previously failed with "Duplicate example_id and reader keys"; a multi-metric input without a choice now names the available metrics.
- Slice retention uses the configured `run.minimum_raw_gap` instead of a fixed 5pp.
- Adapter and scorer files next to `ragscale.yaml` are loaded from their path, not by bare module name. Two projects that both use `adapters.py` in one Python process previously shared whichever module was imported first.
- Loading a configuration no longer adds its directory to `sys.path` more than once.

## 0.5.0, 2026-09-01

- Removed the v1 exploratory layer: `InteractionMatrix`, `quick_audit`, `diagnose`, and the `explore`, `simulate`, and `lossy` commands. They issued verdicts and predictive summaries the frozen protocol does not support. The v1 layer is archived in the paper repository.
- The bundled matrix stays; `load_interaction_matrix()` returns it as a DataFrame.
- Split the runner into `config`, `execute`, `analysis`, `reports`, and `presets` modules. Preset YAML is a plain template, not an escaped f-string. Outputs are unchanged; the committed example reproduces exactly from its call cache.
- CLI now has three commands: `init`, `run`, `replay-audit`.

## 0.4.0, 2026-09-01

- Added key-only presets: `ragscale init --preset openrouter` or `--preset openai` writes a runnable project with 100 bundled HotpotQA rows and hosted compressor and readers. No Python adapters needed.
- Added parallel adapter calls (`run.workers`) and a call checkpoint (`call_cache.jsonl`) so interrupted runs resume without repeating calls.
- Added an OpenAI-compatible model judge under `scoring.metrics` for free-form answers.
- Reports now carry the dataset description, serving provider, per-call cost when the endpoint reports it, and calls made this run.
- Upstream rate limits back off and retry. Empty replies are retried; an empty reader answer scores zero and is flagged, an empty compressor or judge reply raises.
- The default reader prompt asks for the answer span only, because full-sentence answers lose exact match on style rather than content.
- Committed a live OpenRouter run under `examples/openrouter-hotpotqa/` as an illustration, not evidence.

## 0.3.0, 2026-09-01

- Added the YAML-based `init` and `run` workflow for compile-once reader comparisons.
- Added built-in scoring, dataset slices, provenance checks, resource summaries, and portable reports.
- Added OpenAI-compatible compressor and reader adapters with configurable base URLs.
- Scoped every result to the supplied readers and artifacts. No fixed panel size is treated as sufficient for unseen readers.

## 0.2.0, 2026-08-31

- Added `replay-audit` for fixed-artifact reader panels.
- Added artifact, footprint, score, retention, reversal, rescue, and damage checks.
- Bundled the paper's 176,864-row interaction matrix.
- Replaced the proposed three-reader screen after its validation failed.

## 0.1.0

- Initial source-checkout research preview.
