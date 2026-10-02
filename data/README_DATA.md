# Bundled Data

## `hotpotqa_demo_100.jsonl`

The first 100 rows of the paper's frozen HotpotQA distractor slice
(`hotpotqa_distractor_500_v1`), converted to the `ragscale run` record format:
`example_id`, `question`, `reference_answers`, `candidate_pool` (the stored
retrieval candidates with their ids), `question_type`, `difficulty`, `tags`,
and `hints`. The rows are taken in frozen slice order, not selected. The
hosted presets copy this file into a new project. HotpotQA text is
CC BY-SA 4.0.

# Interaction Matrix Schema

`interaction_matrix.csv` and `interaction_matrix.csv.gz` contain one stored row
per benchmark item, reader, and method label. The matrix has 176,864 rows.

This is a broad interaction artifact. It includes auxiliary and sensitivity
conditions beyond the exact fixed-artifact panels used for the paper's main
claims. A common `method` value is not proof that two readers received
byte-identical compressed evidence.

## Columns

| Column | Type | Description |
|---|---|---|
| `dataset` | str | Source benchmark key: `longmemeval`, `hotpotqa`, `musique`, or `nq` |
| `example_id` | str | Item identifier within the stored benchmark slice |
| `question` | str | Question text |
| `question_type` | str | Source or analysis category when available |
| `reference_answer` | str | Stored reference answer |
| `reader_model` | str | Reader model identifier |
| `method` | str | Stored evidence-policy or diagnostic-condition label |
| `correct` | int | Frozen binary outcome, with 1 for correct and 0 for incorrect |
| `judge_score` | float | Stored raw judge score when available |
| `generated_answer` | str | Stored reader output |
| `evidence_tokens` | float | Stored evidence-token count when available |
| `reader_raw_accuracy` | float | Reader's binary score under the stored raw policy, in percent |
| `transition_type` | str | `rescued`, `damaged`, `unchanged_correct`, or `unchanged_wrong` for paired non-raw conditions |

The bundled reference-matrix schema does not contain reader-family identifiers,
candidate-pool hashes, compressed-artifact hashes, native EM/F1 scores, or
analysis-sidecar hashes. Those fields must be joined from the paper artifacts
before running a fixed-artifact audit.

## Transition labels

For one reader and item:

- `rescued`: wrong under the stored raw policy, correct under compression;
- `damaged`: correct under the stored raw policy, wrong under compression;
- `unchanged_correct`: correct under both policies;
- `unchanged_wrong`: wrong under both policies.

Raw-policy rows have an empty `transition_type`. Rescue and damage describe
paired outcome changes. They do not by themselves identify why an answer
changed.

## Dataset scope

| Dataset | Source items | Matrix rows | Paper role |
|---|---:|---:|---|
| LongMemEval | 500 | 79,936 | Semantic-judge replication |
| HotpotQA | 500 | 45,500 | Confirmatory RECOMP panel plus sensitivity policies |
| MuSiQue | 500 | 20,000 | Confirmatory shared-summary panel plus broader stored coverage |
| NQ-Open | 324 | 31,428 | Lexical boundary |

The paper's native-score NQ-Open analysis excludes one malformed source gold
and therefore reports 323 scored items. The source slice and broad binary
matrix retain 324 identifiers where outputs are present.

The public paper count of 176,864 refers only to this interaction matrix. It
does not include the separate deterministic EM/F1 score sidecar, later
TriviaQA robustness panel, or the larger sibling SIEVE workspace.

## Correct use

The matrix supports exploratory inspection of stored reader-policy
interactions. Before making a fixed-compressor claim, an analysis must also
verify:

1. identical benchmark-item footprints across raw and compressed conditions;
2. identical candidate pools and compressed artifacts for every reader;
3. the reader set and reader-family mapping;
4. the metric and score-sidecar version;
5. the resampling and endpoint-selection rule.

The current CSV alone cannot establish these conditions. Use the paper's
exact-footprint manifest and score artifacts for confirmatory reproduction.

## Packaging status

The compressed matrix and reference curves are included in the version 0.3.0
wheel. An installed-wheel smoke test loads all 176,864 rows.
