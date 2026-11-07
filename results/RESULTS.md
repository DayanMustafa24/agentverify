# Phase 1 baseline results (MMFakeBench, official validation set)

> **Historical baseline:** this file records the Phase 1 preliminary results only. They are superseded by the Phase 2 evaluation in
> `results/agent/merged/summary.csv` (150 posts, all methods under the same protocol, same model) and `results/agent/bootstrap.csv`.
> Do not use the Phase 1 numbers below as final project results.

Only numbers copied from real terminal output or saved metrics files belong in this file. Every B1 and B2 number below matches
`results/b1_val30_metrics.json` and `results/b2_val60_metrics.json`.

## Status
- Data pipeline: verified on the 1,000-post official val set (`inspect_dataset.py`: 1000 items, 0 missing images; 300 real / 300 text_false / 100 image_false / 300 mismatch).
- Tests: `pytest -q` and `scripts/smoke_test.py` pass with the offline mock model (mock outputs are not results).
- B1 (CLIP + logistic regression): run on real data (below).
- B2 (single VLM): run on real data with `gemini-3.5-flash-lite`, 60-post stratified val sample, parse_fail_rate = 0.000 (below). One API rate-limit error (429) occurred; the sample is small, so confidence intervals are wide.
- B3: published figure from the base paper only (not run by us).

## Setup actually used
| Item | Value |
|---|---|
| Data | MMFakeBench val only (646 MB download); test set not downloaded in Phase 1 |
| B1 protocol | 700 val posts for training, 300 held-out val posts for evaluation, stratified by 4-way label, seed 42 (`scripts/run_b1_clip_valonly.py`). Superseded in Phase 2 by training on D_train and evaluating on the 150-post val sample. |
| B1 features | frozen CLIP ViT-B/32 (openai/clip-vit-base-patch32): image, text, product, absolute difference, cosine; logistic regression, class-balanced |
| B2 model / endpoint | `gemini-3.5-flash-lite` via Google AI Studio (OpenAI-compatible endpoint), one VLM call per post, no search, no agents |
| B2 protocol | random class-balanced (stratified) sample of 60 official val posts, seed 42 (`scripts/run_b2_single_vlm.py --sample 60`) |

## Results
| Method | n | Binary acc | Binary macro-F1 | 4-way acc | 4-way macro-F1 | ECE (15 bins) | Brier | Parse-fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 CLIP + LR | 300 (held-out val) | 0.817 | 0.785 (95% CI 0.731-0.832) | 0.737 | 0.735 | 0.103 (95% CI 0.075-0.144) | 0.144 | 0.000 |
| B2 gemini-3.5-flash-lite, single VLM call | 60 (val sample) | 0.900 | 0.888 (95% CI 0.795-0.964) | 0.733 | 0.717 | 0.060 (95% CI 0.021-0.140) | 0.083 | 0.000 |
| B3 T2Agent (published, GPT-4o-mini) | 1000 | n/a | 0.631 | n/a | n/a | not reported | not reported | n/a |

B3 is the figure published in the base paper, not a run of ours. B1 (300 posts), B2 (60 posts) and B3 (1,000 posts) were evaluated on different samples, and the backbones differ, so the rows are not directly comparable. B2's n=60 gives wide confidence intervals; treat it as a reference, not a precise estimate. T2Agent's 0.631 is its reported macro-F1 on MMFakeBench and is shown only as a reference point.

### B1 detail
| Metric | Value |
|---|---|
| Binary recall, real / fake | 0.722 / 0.857 |
| Fake precision | 0.878 |
| 4-way recall: real | 0.722 |
| 4-way recall: text_false | 0.900 |
| 4-way recall: image_false | 0.733 |
| 4-way recall: mismatch | 0.589 |
| AURC | 0.077 |
| Selective accuracy at 80% coverage | 0.867 (full-coverage accuracy 0.817) |
| Mean confidence | 0.713 (accuracy 0.817: under-confident on this subset) |

### B2 detail
| Metric | Value |
|---|---|
| Binary recall, real / fake | 0.944 / 0.881 |
| Fake precision | 0.974 |
| Binary macro-F1 95% CI | 0.795-0.964 |
| 4-way recall: real | 0.944 |
| 4-way recall: text_false | 0.667 |
| 4-way recall: image_false | 0.833 |
| 4-way recall: mismatch | 0.556 |
| AURC | 0.026 |
| ECE 95% CI | 0.021-0.140 |
| Selective accuracy at 80% coverage | 0.938 (full-coverage accuracy 0.900) |
| Mean confidence | 0.960 (binary accuracy 0.900: over-confident on this sample) |
| Prompt / completion tokens per post | 1287 / 74 |
| Latency per post | 3.83 s |
| LLM calls / search calls per post | 1.0 / 0.0 |

### B2 run history
- Diagnostic run, n=4 (`results/b2_test.jsonl`): parse_fail 0.000. Used only to check that the endpoint works; not reported as a result.
- First attempt, n=60: INVALID. parse_fail_rate = 1.000, so every reply failed to parse and the metrics only reflected the default "real, 0.5 confidence" fallback (accuracy 0.300, macro-F1 0.231). Not used. The cause was not identified; after error logging was added to `llm.py`, the same endpoint and model id produced valid output.
- Second attempt, n=60: valid (one transient API 503 on post 58, parse_fail_rate 0.000). Its metrics file was overwritten by the rerun below, so none of its numbers are reported here.
- Latest rerun, n=60: VALID, `gemini-3.5-flash-lite`, one API 429 rate-limit error logged, parse_fail_rate 0.000, all 60 posts in the metrics. This is the run reported above and it matches `results/b2_val60_metrics.json`.

## Observations (to be tested in Phase 2, not conclusions)
- A trained CLIP classifier scores well on this subset. Because it was trained on part of the same val set, its strength may come from surface cues of how MMFakeBench was built (for example text style or image source) rather than from checking facts. This was not tested; it is one reason Phase 2 trains B1 on D_train and also uses the perturbation suite.
- Mismatch is the weakest 4-way class for both baselines (recall 0.589 for B1 and 0.556 for B2), which is where evidence and caption-image consistency checks should matter.
- B1 is under-confident (mean confidence 0.713 vs accuracy 0.817, ECE 0.103) and B2 is over-confident (mean confidence 0.960 vs accuracy 0.900, ECE 0.060). Calibration is not reliable in either direction, which motivates Gap 1. B2's ECE comes from only 60 posts (95% CI 0.021-0.140), so it is not evidence that B2 is better calibrated than B1.

## Notes
- The full D_train / D_cal / D_test protocol from the test set is applied in Phase 2.
- Confidence for B1 is the maximum class probability; B2 uses the model's verbalised confidence.
- B2 was run through a hosted API (Google AI Studio free tier), so val images and captions were sent to an external service.