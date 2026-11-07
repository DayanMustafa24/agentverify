# AgentVerify (Phase 2: calibrated, evidence-grounded multi-agent fake-news detection)

Agentic multimodal fake news detection from text and images.
Course: AI4015 Agentic AI. Project code G1-12. Supervisor: Muhammad Sajjad.

Phase 1 delivered the data pipeline, shared contracts, model provider layer, evidence-tool
interface, perturbation utilities, evaluation code (accuracy, macro-F1, ECE, Brier, risk-coverage,
cost) and two non-agentic baselines. Phase 2 builds the four agents (text, image, evidence, decision),
the calibration layer, the robustness suite and the ablation runs on top of that foundation.

![architecture](docs/architecture.png)

| Baseline | What it is |
|---|---|
| B1 | Frozen CLIP embeddings + logistic regression |
| B2 | One vision-language-model call per post, no tools, no agents |
| B3 | Base paper (T2Agent) / MMD-Agent, compared through published numbers on the MMFakeBench val set (not re-run by us) |

## Changes since Phase 1

| Item | Phase 1 proposal | Now | Reason |
|---|---|---|---|
| B2 backbone | Gemini 2.5 Flash | `gemini-3.5-flash-lite` via Google AI Studio (OpenAI-compatible endpoint) | TODO: state the real reason (availability / free-tier quota) |
| B2 evaluation set | 1,000 official val posts | 60-post stratified val sample (seed 42); full-val run pending | free-tier rate limits (one 429 and one 503 seen) |
| B1 protocol | trained on D_train (2,000 posts from test) | Phase 1 numbers use 700 val / 300 val (val only; test set not yet downloaded) | test set not downloaded in Phase 1; **to be re-run on D_train with no overlap with D_val** |
| Work division | "Worked locally" | see Section 7 | TA feedback |

The base paper (T2Agent) and Gaps 1-3 are unchanged.

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install -e ".[clip]"        # only needed for B1 (torch + transformers)
cp .env.example .env            # then put your key in .env and export it
```
Any OpenAI-compatible endpoint works. For a free local run use Ollama and set
`OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama`, `AV_MODEL=qwen2.5vl:7b`.

To reproduce the reported B2 run, use the Gemini OpenAI-compatible endpoint
(never commit your key; set it only in your shell or in `.env`, which is git-ignored):
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
$env:AV_MODEL="gemini-3.5-flash-lite"
```
Note: B2 sends val images and captions to an external service (Google AI Studio free tier).

## 2. Get the data (MMFakeBench, gated)
1. Open https://huggingface.co/datasets/liuxuannan/MMFakeBench, log in, accept the usage terms.
2. Download and unzip so you have `data/raw/MMFakeBench/MMFakeBench_val/` and `.../MMFakeBench_test/`.
3. **Never commit the data** (licence forbids redistribution; `data/raw/` is git-ignored).
```bash
python scripts/inspect_dataset.py --root data/raw/MMFakeBench   # check labels + images exist
python scripts/make_splits.py   --root data/raw/MMFakeBench     # writes id lists to data/splits/
```
If `inspect_dataset.py` reports an unknown `fake_cls` string, extend `normalize_source()` in
`src/agentverify/data/mmfakebench.py`.

Planned splits (stratified, seed 42, stored as id lists only):

| Split | Size | Use |
|---|---|---|
| D_val | 1,000 (official val) | main results, comparison with T2Agent and MIRAGE |
| D_train | 2,000 from test | fit B1 only |
| D_cal | 1,000 from test | fit the calibrator (Gap 1) |
| D_test | 2,000 from test | final held-out evaluation |
| D_rob | 300 from test | perturbation runs (Gap 2) |

Phase 1 baseline runs used only the validation set (test set not yet downloaded). See "Changes since Phase 1".

## 3. Verify the install (no key, no data needed)
```bash
pytest -q                       # 15 passed
python scripts/smoke_test.py
```
Smoke-test numbers use a mock model and are **not** results.

## 4. Run the baselines
```bash
python scripts/run_b2_single_vlm.py --root data/raw/MMFakeBench --split val --sample 4  --out results/b2_test.jsonl    # quick diagnostic (checks parse_fail and API errors)
python scripts/run_b2_single_vlm.py --root data/raw/MMFakeBench --split val --sample 60 --out results/b2_val60.jsonl   # reported run
python scripts/run_b1_clip_valonly.py --root data/raw/MMFakeBench                                                      # reported B1 run (700 train / 300 eval, val only)
```
<!-- TODO: confirm the exact B1 arguments/output path, and whether run_b1_clip.py is still used. -->
Each run writes `*.jsonl` (one RunRecord per post), `*_metrics.json` and `*_reliability.png`.
Parse failures are counted as wrong and reported as `parse_fail_rate`.
A B2 run is only valid if `parse_fail_rate` is close to 0. Look for `[llm error]` lines in the
console output (404 = wrong model id, 401/403 = bad key, 429 = rate limit, 503 = temporary overload).
A 4-post diagnostic run is not a result and must not be reported.

## 4b. Baseline results (MMFakeBench official val)

| Method | n | Binary acc | Binary macro-F1 | 4-way acc | 4-way macro-F1 | ECE (15 bins) | Brier | Parse-fail |
|---|---|---|---|---|---|---|---|---|
| B1 CLIP + LR | 300 (held-out val) | 0.817 | 0.785 (95% CI 0.731-0.832) | 0.737 | 0.735 | 0.103 (CI 0.075-0.144) | 0.144 | 0.000 |
| B2 Gemini 3.5 Flash-Lite, single call | 60 (val sample) | 0.900 | 0.888 (95% CI 0.795-0.964) | 0.733 | 0.717 | 0.060 (CI 0.021-0.140) | 0.083 | 0.000 |
| B3 T2Agent (GPT-4o-mini), **taken from the paper, not run by us** | 1,000 | n/a | 0.631 | n/a | n/a | not reported | not reported | n/a |

B1 (300 posts), B2 (60 posts) and B3 (1,000 posts) were evaluated on different subsets, so the rows are
**not directly comparable**. T2Agent uses a different backbone (GPT-4o-mini) than ours; this is a limitation
of the comparison. B2's n=60 gives wide confidence intervals, so treat it as a baseline reference only.
Full detail and the B2 run history (including one invalid run that was discarded) are in `results/Result.md`.

**B2 detail.** Binary recall real / fake 0.944 / 0.881, fake precision 0.974. 4-way recall:
real 0.944, text_false 0.667, image_false 0.833, mismatch 0.556. AURC 0.026, selective accuracy
at 80% coverage 0.938 (full coverage 0.900). Mean confidence 0.960 vs accuracy 0.900 (over-confident).
Cost per post: about 1,287 prompt and 74 completion tokens, 3.83 s, 1 LLM call, 0 search calls.
One 503 and one 429 occurred during the valid runs; parse_fail stayed at 0.000.

**B1 detail.** Binary recall real / fake 0.722 / 0.857, fake precision 0.878. 4-way recall:
real 0.722, text_false 0.900, image_false 0.733, mismatch 0.589. AURC 0.077, selective accuracy
at 80% coverage 0.867 (full 0.817). Mean confidence 0.713 vs accuracy 0.817 (under-confident).

**Observations (hypotheses for Phase 2, not conclusions).**
- Mismatch is the weakest 4-way class for both baselines, which is where evidence and caption-image checks should help.
- B1 is under-confident and B2 over-confident, so raw confidence is unreliable in both directions (motivates Gap 1). B2's ECE comes from 60 posts and is not evidence that B2 is better calibrated than B1.
- B1 is trained on part of val, so its score may reflect dataset-construction cues. This is untested.

## 5. Layout
```
src/agentverify/
  contracts.py      Pydantic state: PostInput, AgentVerdict, EvidenceItem, FinalVerdict, CostLog, RunRecord
  llm.py            OpenAI-compatible client + offline MockClient
  prompts.py        B2 prompt
  data/             MMFakeBench loader, fixed splits, synthetic test data
  baselines/        clip_lr.py (B1), single_vlm.py (B2)
  tools/search.py   Wikipedia + DuckDuckGo evidence tool, trusted-domain rerank, disk cache
  perturb.py        label-preserving image/text perturbations (Phase 2, Gap 2)
  eval/             metrics.py (acc, F1, ECE, Brier, AURC, bootstrap CI), report.py
scripts/ tests/ docs/ results/
```
<!-- TODO: add the Phase 2 modules (agents/, orchestrator, calibration layer, ablation switches) once they exist. -->

## 6. Gap-resolution table (filled in with E4 results)

| Gap | Method / component | Metric | Base paper result | Our result | Addressed? |
|---|---|---|---|---|---|
| Gap 1: calibration | Calibration layer + abstention | ECE, Brier, AURC, accuracy at 80% coverage | not reported (T2Agent) | pending | pending |
| Gap 2: robustness | Seeded label-preserving perturbation suite | delta macro-F1, delta ECE, label-flip rate at severity 3 | not reported | pending | pending |
| Gap 3: agent contribution and cost | Leave-one-agent-out switches + cost logger | per-class recall, macro-F1, USD per 1k posts | ablates search/tools only; about 5x cost | pending | pending |

## 6b. Experiments

| ID | What | Command | Output | Status |
|---|---|---|---|---|
| E1 | B1 baseline | `python scripts/run_b1_clip_valonly.py` | `results/` | done on val only; re-run on D_train |
| E2 | Base paper | published numbers only (not re-implemented) | n/a | taken from paper |
| E3 | Full agentic system | TODO | TODO | pending |
| E4 | Gap-wise experiments | TODO | TODO | pending |
| E5 | Ablation (one component removed at a time) | TODO | TODO | pending |
| E6 | Cost, latency, reliability | TODO | TODO | pending |

## 6c. Status
- [x] pipeline, contracts, provider layer, evidence tool, perturbations, metrics, tests (15 pass)
- [x] B1 and B2 implemented; offline smoke test passes
- [x] B1 on 300 held-out val posts (val-only protocol)
- [x] B2 on a 60-post stratified val sample (Gemini 3.5 Flash-Lite, parse_fail = 0.000)
- [ ] B2 on the full 1,000-post val set
- [ ] B1 re-run on D_train / D_val with no overlap
- [ ] hand-check a sample of perturbations to confirm labels are preserved (for example crops that remove the manipulated region)
- [ ] text, image, evidence and decision agents, calibration layer, robustness and ablation runs
- [ ] T2Agent comparison on a small GPT-4o-mini subset

## 7. Team responsibilities

| Member | Roll no. | Responsibilities |
|---|---|---|
| Dayyan Mustafa | 23i-0702 | TODO |
| Muhammad Adeel | 23i-0739 | TODO |
| Abdul Hadi | 23i-0707 | TODO |

Each member commits their own work from their own GitHub account.