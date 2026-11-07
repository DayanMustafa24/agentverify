# Phase 1 baseline results (MMFakeBench, official validation set)

Only numbers copied from real terminal output / metrics files belong in this file.

## Status
- Data pipeline: verified on the 1,000-post official val set (`inspect_dataset.py`: 1000 items, 0 missing images; 300 real / 300 text_false / 100 image_false / 300 mismatch).
- Tests: `pytest -q` and `scripts/smoke_test.py` pass with the offline mock model (mock outputs are NOT results).
- B1 (CLIP + logistic regression): RUN on real data (below).
- B2 (single VLM): RUN on real data with Gemini 3.5 Flash-Lite, 60-post stratified val sample, parse_fail_rate = 0.000 (below). One API rate-limit error (429) occurred during the run.The sample is small, so confidence intervals are wide.
- B3: published figure from the base paper only (not run by us).

## Setup actually used
| Item | Value |
|---|---|
| Data | MMFakeBench val only (646 MB download); test set not downloaded in Phase 1 |
| B1 protocol | 700 val posts for training, 300 held-out val posts for evaluation, stratified by 4-way label, seed 42 (`scripts/run_b1_clip_valonly.py`) |
| B1 features | frozen CLIP ViT-B/32 (openai/clip-vit-base-patch32): image, text, product, absolute difference, cosine; logistic regression, class-balanced |
| B2 model / endpoint | `gemini-3.5-flash-lite` via Google AI Studio (OpenAI-compatible endpoint), one VLM call per post, no search, no agents |
| B2 protocol | random class-balanced (stratified) sample of 60 official val posts, seed 42 (`scripts/run_b2_single_vlm.py --sample 60`) |

## Results
| Method | n | Binary acc | Binary macro-F1 | 4-way acc | 4-way macro-F1 | ECE (15 bins) | Brier | Parse-fail |
|---|---|---|---|---|---|---|---|---|
| B1 CLIP + LR | 300 (held-out val) | 0.817 | 0.785 (95% CI 0.731-0.832) | 0.737 | 0.735 | 0.103 (95% CI 0.075-0.144) | 0.144 | 0.0 |
| B2 Gemini 3.5 Flash-Lite, single VLM call | 60 (val sample) | 0.900 | 0.888 (95% CI 0.795-0.964) | 0.733 | 0.717 | 0.060 (95% CI 0.021-0.140) | 0.083 | 0.000 |
| B3 T2Agent (published, GPT-4o-mini) | 1000 | n/a | 0.631 | n/a | n/a | not reported | not reported | n/a |

B3 is the published figure from the base paper, not a run of ours. B1 (300 posts) and B2 (60 posts) were evaluated on different subsets of val, and B3 on all 1,000, so the three rows are not directly comparable. B2's n=60 gives wide confidence intervals, so treat it as a baseline reference, not a precise estimate. T2Agent's 0.631 is its reported macro-F1 on MMFakeBench; it is shown as a reference point only.

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
| Mean confidence | 0.713 (accuracy 0.817, so B1 is under-confident on this subset) |

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
| Mean confidence | 0.960 (binary accuracy 0.900, so B2 is over-confident on this sample) |
| Prompt / completion tokens per post | 1287 / 74 |
| Latency per post | 3.83 s |
| LLM calls / search calls per post | 1.0 / 0.0 |
### B2 run history
- Diagnostic run, n=4 (`results/b2_test.jsonl`): parse_fail 0.000. Used only to check that the endpoint works; not reported as a result.
- First attempt, n=60: INVALID. parse_fail_rate = 1.000, so every reply failed to parse and the metrics only reflected the default "real, 0.5 confidence" fallback (accuracy 0.300, macro-F1 0.231). Not used. The cause was not identified; after the error logging was added to `llm.py`, the same endpoint and model id produced valid output.
- Second attempt, n=60: VALID, reported above.
- During the valid run, one request (post 58) returned a transient API 503 ("model is currently experiencing high demand"); parse_fail_rate stayed at 0.000.
- Latest rerun, n=60: VALID, Gemini 3.5 Flash-Lite. One API 429 rate-limit/quota error was logged during execution. Final parse_fail_rate = 0.000, and all 60 samples are represented in the metrics. The run completed successfully, but the rate-limit event is documented.

## Observations (to be tested in Phase 2, not conclusions)
- A trained CLIP classifier scores well on this subset. Because it is trained on part of the same val set, its strength may come from surface cues of how MMFakeBench was built (e.g. text style, image source) rather than checking facts. We have not tested this; it is a reason to also use the test-derived splits and the perturbation suite in Phase 2.
- Mismatch is the weakest 4-way class for both baselines (recall 0.589 for B1 and 0.444 for B2), which is where evidence and caption-image consistency checks should matter.
- B1 is under-confident (mean confidence 0.713 vs accuracy 0.817, ECE 0.103) and B2 is over-confident (mean confidence 0.960 vs accuracy 0.900, ECE 0.060). Calibration is not reliable in either direction, which motivates Gap 1. B2's ECE is from only 60 posts (95% CI 0.014-0.138), so it is not evidence that B2 is better calibrated than B1.

## Notes
- The full D_train / D_cal / D_test protocol from the test set is applied in Phase 2.
- Confidence for B1 is the maximum class probability; B2 uses the model's verbalised confidence.
- B2 was run through a hosted API (Google AI Studio free tier), so val images and captions were sent to an external service.