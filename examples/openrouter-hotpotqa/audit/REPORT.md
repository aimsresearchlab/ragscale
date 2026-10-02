# ragscale reader-upgrade audit

Dataset: **hotpotqa-demo-100**  
Comparison: **qwen/qwen-2.5-7b-instruct → meta-llama/llama-3.3-70b-instruct**  
Examples: **100**

Illustration only. First 100 rows of the paper's frozen HotpotQA distractor slice, one fresh LLM-summary artifact per row, two hosted readers. Not paper evidence and not a verdict about other readers or compressors.

## Decision summary

| Metric | Current compression gain | Raw reader upgrade | Visible compressed upgrade | Upgrade retention | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| exact_match | 11.0pp | 9.0pp | 3.0pp | 33.3% | observed-upgrade-attenuated |
| token_f1 | 11.3pp | 9.5pp | 4.4pp | 46.5% | observed-upgrade-attenuated |
| judge_match | 13.0pp | 12.0pp | 4.0pp | 33.3% | observed-upgrade-attenuated |

## exact_match detail

- Raw reader upgrade interval: -0.0pp to 18.0pp
- Visible compressed upgrade interval: -5.0pp to 11.0pp
- Upgrade-retention interval: -50.0% to 116.7%
- Configured retention alert: 75%
- Current reader rescue: 16.0%; raw-correct damage: 9.4%
- Candidate reader rescue: 8.0%; raw-correct damage: 4.8%

## token_f1 detail

- Raw reader upgrade interval: 1.4pp to 18.3pp
- Visible compressed upgrade interval: -1.0pp to 10.3pp
- Upgrade-retention interval: -9.7% to 130.5%
- Configured retention alert: 75%

## judge_match detail

- Raw reader upgrade interval: 3.0pp to 21.0pp
- Visible compressed upgrade interval: -2.0pp to 10.0pp
- Upgrade-retention interval: -23.1% to 100.0%
- Configured retention alert: 75%
- Current reader rescue: 17.0%; raw-correct damage: 5.3%
- Candidate reader rescue: 7.0%; raw-correct damage: 2.3%

The deployment and reader decisions are separate. Compression gain measures the current complete pipeline. Upgrade retention measures how much of the observed current-to-candidate reader upgrade remains under the deployed compressor.

## Provenance

- Candidate pools: 100
- Compile-once artifact rows: 100
- Unique compressed content hashes: 100
- Exact reader footprints: true
- Raw policy: `hotpotqa-distractor-10`

## Dataset hints

- source: hotpotqa_distractor_500_v1
- language: en

## Question and dataset slices

| Metric | Hint | Value | Rows | Raw upgrade | Compressed upgrade | Retention |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| exact_match | question_type | bridge | 82 | 13.4pp | 4.9pp | 36.4% |
| exact_match | question_type | comparison | 18 | -11.1pp | -5.6pp | 50.0% |
| exact_match | difficulty | hard | 100 | 9.0pp | 3.0pp | 33.3% |
| token_f1 | question_type | bridge | 82 | 12.5pp | 5.2pp | 41.1% |
| token_f1 | question_type | comparison | 18 | -4.6pp | 0.9pp | not admissible |
| token_f1 | difficulty | hard | 100 | 9.5pp | 4.4pp | 46.5% |
| judge_match | question_type | bridge | 82 | 13.4pp | 3.7pp | 27.3% |
| judge_match | question_type | comparison | 18 | 5.6pp | 5.6pp | 100.0% |
| judge_match | difficulty | hard | 100 | 12.0pp | 4.0pp | 33.3% |

## Changed examples

| Example | Type | Current reader | Candidate reader |
| --- | --- | --- | --- |
| hpqa-0003 | bridge | unchanged | damaged |
| hpqa-0004 | bridge | rescued | unchanged |
| hpqa-0008 | bridge | damaged | unchanged |
| hpqa-0010 | bridge | rescued | rescued |
| hpqa-0012 | bridge | rescued | rescued |
| hpqa-0019 | bridge | rescued | unchanged |
| hpqa-0022 | bridge | rescued | unchanged |
| hpqa-0024 | bridge | unchanged | damaged |
| hpqa-0026 | bridge | rescued | unchanged |
| hpqa-0042 | bridge | rescued | unchanged |

## Resources

- Mean compressor latency: 4586.7 ms
- Mean raw reader latency: 795.7 ms
- Mean compressed reader latency: 598.8 ms
- Adapter-reported cost: $0.0680
- Compressor calls this run: 100; reader calls this run: 400

## Scope

This report describes the supplied readers, rows, evidence policy, artifacts, and metrics. It does not certify unseen readers or future compressor draws.
