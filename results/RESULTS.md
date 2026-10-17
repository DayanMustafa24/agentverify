# Phase 1 baseline results (MMFakeBench, official validation set)

Only numbers copied from real terminal output / metrics files belong in this file. A cell stays `not run` until it is.

## Status
- Data pipeline: verified on the 1,000-post official val set (`inspect_dataset.py`: 1000 items, 0 missing images; 300 real / 300 text_false / 100 image_false / 300 mismatch).
- Tests: `pytest -q` and `scripts/smoke_test.py` pass with the offline mock model (mock outputs are NOT results).
- B1 (CLIP + logistic regression): RUN on real data (below).
- B2 (single VLM): first attempt (Gemini 2.5 Flash, 60-post stratified sample) FAILED: parse_fail_rate = 1.000, i.e. no reply could be parsed, so its accuracy/F1 (0.300 / 0.231) only reflect the default "real, 0.5 confidence" fallback and are NOT a valid result. Cause being diagnosed (API error output now printed by `llm.py`). Row stays `not run` until a run with parse_fail near 0 completes.

## Setup actually used
| Item | Value |
|---|---|
| Data | MMFakeBench val only (646 MB download); test set not downloaded in Phase 1 |
| B1 protocol | 700 val posts for training, 300 held-out val posts for evaluation, stratified by 4-way label, seed 42 (`scripts/run_b1_clip_valonly.py`) |
| B1 features | frozen CLIP ViT-B/32 (openai/clip-vit-base-patch32): image, text, product, absolute difference, cosine; logistic regression, class-balanced |
| B2 model / endpoint | Gemini 2.5 Flash via Google AI Studio (OpenAI-compatible endpoint); first attempt invalid, see Status |

## Results
| Method | n | Binary acc | Binary macro-F1 | 4-way acc | 4-way macro-F1 | ECE (15 bins) | Brier | Parse-fail |
|---|---|---|---|---|---|---|---|---|
| B1 CLIP + LR | 300 (held-out val) | 0.817 | 0.785 (95% CI 0.731-0.832) | 0.737 | 0.735 | 0.103 (95% CI 0.075-0.144) | 0.144 | 0.0 |
| B2 single VLM | not run (first attempt invalid, parse_fail 1.0) | | | | | | | |
| B3 T2Agent (published, GPT-4o-mini) | 1000 | n/a | 0.631 | n/a | n/a | not reported | not reported | n/a |

B3 is the published figure from the base paper, not a run of ours. B1 was evaluated on a 300-post subset of val and B3 on all 1,000, so the numbers are not directly comparable. T2Agent's 0.631 is its reported macro-F1 on MMFakeBench; it is shown as a reference point only.

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

## Observations (to be tested in Phase 2, not conclusions)
- A trained CLIP classifier scores well on this subset. Because it is trained on part of the same val set, its strength may come from surface cues of how MMFakeBench was built (e.g. text style, image source) rather than checking facts. We have not tested this; it is a reason to also use the test-derived splits and the perturbation suite in Phase 2.
- Mismatch is the weakest class (recall 0.589), which is where evidence and caption-image consistency checks should matter.
- Confidence is not well matched to accuracy (ECE 0.103), which motivates Gap 1.

## Notes
- The full D_train / D_cal / D_test protocol from the test set is applied in Phase 2.
- Confidence for B1 is the maximum class probability; B2 will use the model's verbalised confidence.
