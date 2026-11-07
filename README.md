# AgentVerify: agentic multimodal fake-news detection from text and images

AI4015 Agentic AI, Phase 2. Project code **G1-12**. Supervisor: Muhammad Sajjad.
Base paper: T2Agent (AAAI 2026). Benchmark: MMFakeBench (text + image posts, 4 classes: real, text_false, image_false, mismatch).

AgentVerify replaces a single fixed classifier with four cooperating agents: a **text agent** (reads the caption), an **image agent**
(manipulation and caption-image consistency), an **evidence agent** (Wikipedia + web retrieval, judges the claim against the sources) and a
**decision agent** (fuses the reports into a verdict, a 4-way label, a confidence and a short explanation). A calibration layer rescales the
confidence, and every agent can be switched off for ablation.

![architecture](docs/architecture.png)

## Headline result (honest summary)
On a stratified 150-post sample of the official MMFakeBench validation set, the four-agent system matched, but did not beat, the single-VLM
baseline (B2) and a CLIP + logistic-regression classifier trained on 2,000 benchmark posts (B1). 4-way macro-F1 was 0.672 (agents), 0.667 (B2)
and 0.752 (B1); the paired bootstrap difference between the agents and B1 is -0.080 (95% CI [-0.193, 0.035]), so the three are statistically
indistinguishable at this sample size. The agents used about 1.9x B2's tokens and 2.9x its latency. The image agent carries most of the signal,
the text agent adds a small 4-way gain, and the evidence agent and Platt calibration showed no measurable benefit. Under caption typos the full
system changed its binary verdict more often than B2 (20% vs 5% of 60 posts).

## 0. Changes since Phase 1
| Item | Phase 1 proposal | Now | Reason |
|---|---|---|---|
| Backbone | GPT-4o-mini and Qwen2.5-VL-7B (Gemini 2.5 Flash used for the first B2 run) | `gemini-3.5-flash-lite` via Google AI Studio (OpenAI-compatible endpoint) for B2 and all agents | Gemini 2.5 Flash showed 20 requests/day on our free tier (5 RPM); 3.5 Flash-Lite showed 500 requests/day (15 RPM). No paid API was used. |
| B2 | 60-post sample, Gemini 2.5 Flash | re-run on the same 150 posts as the agents, with the same model | all methods must share backbone and evaluation set |
| B1 | 700 / 300 split of val (Phase 1 preliminary, superseded) | trained on D_train (2,000 test-derived posts), evaluated on the same 150 val posts | removes the train/val overlap raised in TA feedback |
| Evaluation size | full val (1,000) | 150 stratified val posts (seed 42); calibration set 60 stratified test-split posts | free-tier request limits |
| Repeated runs | 3 runs, mean and std | not done | temperature 0 plus a disk cache makes repeats identical; we report bootstrap 95% CIs over posts instead |
| T2Agent comparison on GPT-4o-mini | small subset planned | not done (no paid API) | stated as a limitation |
| Robustness (Gap 2) | perturbation suite | see Section 6 (status in the gap table) | |

The base paper and Gaps 1-3 are unchanged.

## 1. Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install -e ".[clip]"          # only for B1 (torch + transformers)
$env:PYTHONUTF8="1"               # Windows: avoids cp1252 encoding errors when writing jsonl
```
Configuration is read from **environment variables** (the code does not load `.env` by itself). Set them in the same shell:
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"                    # never commit a key
$env:OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
$env:AV_MODEL="gemini-3.5-flash-lite"             # exact model id shown in Google AI Studio
$env:AV_TEMPERATURE="0"
```
Any OpenAI-compatible endpoint works (OpenAI, Ollama, vLLM). `.env.example` lists the variable names.
Note: with the hosted API, validation images and captions are sent to an external service (Google AI Studio free tier).

## 2. Data (MMFakeBench, gated)
1. Open https://huggingface.co/datasets/liuxuannan/MMFakeBench, log in, accept the terms.
2. Unzip to `data/raw/MMFakeBench/MMFakeBench_val/` and `data/raw/MMFakeBench/MMFakeBench_test/` (each with `source/`, `real/`, `fake/`).
3. The licence forbids redistribution: **raw data is never committed**.
```powershell
python scripts/inspect_dataset.py --root data/raw/MMFakeBench
python scripts/make_splits.py   --root data/raw/MMFakeBench      # id lists in data/splits/
```
| Split | Source | Use |
|---|---|---|
| D_val | official val (1,000) | main evaluation (150-post stratified sample, seed 42) |
| D_train | 2,000 test-derived | fits B1 only |
| D_cal | test-derived | calibration (we sample 60 stratified test-split posts, so it never overlaps D_val) |
| D_rob | 300 test-derived | perturbation study |

## 2b. Verify the install (no key, no data)
```powershell
pytest -q
python scripts/smoke_test.py
python scripts/smoke_agents.py
```
These use a mock model; their outputs are **not** results.

## 3. Reproduce the experiments
Runs are cached on disk (`data/cache/`) and resumable. Three members run one shard each (`0/3`, `1/3`, `2/3`) with their own key.
```powershell
# calibration set (full config only), then evaluation (B2, full system, three ablations)
python scripts/run_agent.py --root data/raw/MMFakeBench --role cal  --n 60  --shard 0/3 --configs full
python scripts/run_agent.py --root data/raw/MMFakeBench --role eval --n 150 --shard 0/3 --configs b2 full no_text no_image no_evidence
# repeat with --shard 1/3 and --shard 2/3, put the three results/agent/shardN folders together, then:
python scripts/merge_eval.py --shards results/agent/shard0 results/agent/shard1 results/agent/shard2   # -> results/agent/merged/summary.csv
python scripts/b1_on_eval_set.py --root data/raw/MMFakeBench --n 150                                   # B1 on the same 150 posts
python scripts/bootstrap_compare.py                                                                    # -> results/agent/bootstrap.csv
python scripts/failure_analysis.py                                                                     # -> failure_categories.csv, failure_cases.md
# robustness (Gap 2)
python scripts/run_robustness.py --root data/raw/MMFakeBench --n 60 --shard 0/3 --save-examples 10
python scripts/merge_robust.py --shards results/robust/shard0 results/robust/shard1 results/robust/shard2
```
`--max-calls` (default 450) stops a run before the daily request quota; rerun the same command later and cached calls are skipped.

**Exact reproduction without any API calls.** Web search results change over time, so we provide a snapshot of the caches:
Download `cache_snapshot.zip` from the GitHub Releases page for this repository., unzip it into `data/cache/`, and rerun the commands with the same
model id; every call is served from the cache. Experiment map: E1 = B1 and B2 rows, E2 = published T2Agent numbers (not re-run), E3 = `full`,
E4 = gap table (calibration: `full_cal`; robustness: `results/robust`), E5 = `no_*` rows, E6 = cost columns.

## 4. Results (150 stratified val posts, one run, temperature 0)
| Config | n | Bin. acc | Bin. macro-F1 | 4-way macro-F1 | ECE | Brier | Tokens/post | LLM calls | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| B1 CLIP + LR (D_train) | 150 | 0.847 | 0.828 | 0.752 | 0.107 | 0.136 | n/a | 0 | n/a |
| B2 single VLM | 150 | 0.867 | 0.850 | 0.667 | 0.096 | 0.120 | 1361 | 1.0 | 3.4 |
| Full system | 150 | 0.847 | 0.827 | 0.672 | 0.102 | 0.133 | 2558 | 4.97 | 9.8 |
| no text agent | 150 | 0.853 | 0.837 | 0.637 | 0.103 | 0.131 | 2332 | 3.99 | 7.5 |
| no image agent | 150 | 0.593 | 0.592 | 0.344 | 0.318 | 0.328 | 1204 | 3.99 | 7.4 |
| no evidence agent | 150 | 0.853 | 0.828 | 0.656 | 0.077 | 0.122 | 1915 | 3.0 | 6.5 |
| Full + Platt calibration | 150 | 0.847 | 0.827 | 0.672 | 0.106 | 0.132 | 2558 | 4.97 | 9.8 |

Search calls per post: 1.76 for the full system. The ablation rows show fewer search calls only because searches were already cached when they ran.
Source: `results/agent/merged/summary.csv`. T2Agent's published 4-way macro-F1 of 0.631 (GPT-4o-mini, full val set) is **taken from the paper, not reproduced**;
backbone and sample differ, so it is an indicative reference only.

**Paired bootstrap over posts (2,000 resamples, 95% CI of the difference A minus B):**
| A vs B | Metric | Difference | 95% CI |
|---|---|---|---|
| full vs B2 | 4-way macro-F1 | +0.005 | [-0.060, 0.069] |
| full vs B2 | binary macro-F1 | -0.023 | [-0.077, 0.029] |
| full vs no_text | 4-way macro-F1 | +0.036 | [0.001, 0.073] |
| full vs no_evidence | 4-way macro-F1 | +0.017 | [-0.006, 0.044] |
| full vs no_image | accuracy | +0.253 | [0.180, 0.327] |
| full_cal vs full | ECE | +0.004 | [-0.057, 0.064] |
| full vs B1 | 4-way macro-F1 | -0.080 | [-0.193, 0.035] |
| full vs B1 | binary macro-F1 | -0.002 | [-0.082, 0.076] |
| B2 vs B1 | 4-way macro-F1 | -0.085 | [-0.205, 0.034] |
## 5. Failure analysis
The full system made 46 four-way errors on 150 posts (30.7%): 27 where no specialist had the right label and 19 where the decision agent overrode a
correct specialist. Most frequent: mismatch predicted as real (11), mismatch predicted as image_false (8), text_false predicted as image_false (8).
Two success and two failure trajectories are in `results/agent/failure_cases.md`; category counts are in `results/agent/failure_categories.csv`.
Sample traces: `logs/sample_traces/`.

## 6. Gap-resolution table
| Gap | Method / component | Metric | Base paper result | Our result | Addressed? |
|---|---|---|---|---|---|
| 1. Calibration | Platt scaling of the verdict confidence (60-post calibration set) | ECE, Brier | not reported (T2Agent) | ECE 0.106 vs 0.102 raw (diff +0.004, CI [-0.057, 0.064]); B2 0.096 | **Not** |
| 2. Robustness | Seeded JPEG and caption-typo perturbations | Binary/4-way macro-F1, ECE, prediction flip rate | not reported | n=60: full-system typo flip rate 20% vs B2 5%; full binary F1 change -0.073 vs B2 +0.049. JPEG flip rates 8.3% vs 6.7%. | Partially |

**Robustness results (n=60 per condition, one run)**

| Config | Condition | Binary macro-F1 | 4-way macro-F1 | ECE | Binary flip vs clean | 4-way flip vs clean |
|---|---|---:|---:|---:|---:|---:|
| B2 | Clean | 0.821 | 0.537 | 0.131 | 0% | 0% |
| B2 | JPEG3 | 0.817 | 0.571 | 0.122 | 6.7% | 11.7% |
| B2 | Typos3 | 0.870 | 0.533 | 0.075 | 5% | 11.7% |
| Full | Clean | 0.817 | 0.539 | 0.133 | 0% | 0% |
| Full | JPEG3 | 0.791 | 0.594 | 0.121 | 8.3% | 11.7% |
| Full | Typos3 | 0.744 | 0.474 | 0.125 | 20% | 26.7% |
The robustness experiment used one run on 60 test-derived posts (D_rob), separate from the 150-post validation sample (D_val). These results are suggestive and should not be treated as conclusive evidence of general robustness.
| 3. Agent contribution and cost | leave-one-agent-out switches and a cost log | per-class recall, macro-F1, tokens, latency | ablates search/tools only; about 5x cost | image agent dominant (4-way F1 -0.328 without it); text +0.036 (CI [0.001, 0.073]); evidence not significant; 1.9x tokens and 2.9x latency vs B2 | **Fully** (analysis; no accuracy gain over B2) |

## 7. Repository layout
```
src/agentverify/
  contracts.py        shared state: PostInput, AgentVerdict, EvidenceItem, FinalVerdict, CostLog, RunRecord
  llm.py, config.py   OpenAI-compatible client, mock client, env config
  prompts.py          B2 prompt
  agents/             text_agent, image_agent, evidence_agent, decision_agent, orchestrator (ablation switches), common (cache, rate limit)
  calibration.py      Platt scaling
  robustness.py       label-safe perturbed copies of posts (new post_id)
  perturb.py          image / text perturbations
  tools/search.py     Wikipedia + DuckDuckGo evidence tool with trusted-domain rerank and disk cache
  baselines/          clip_lr.py (B1), single_vlm.py (B2)
  data/, eval/        loader and splits; metrics (acc, F1, ECE, Brier, AURC, bootstrap CI)
scripts/ tests/ docs/ results/ logs/
```
`docs/devlog.md` records each failure, change and measured effect. `docs/STATE_AND_CONTRACTS.md` describes the state and agent contracts.

## 8. Team responsibilities
| Member | Roll no. | Responsibilities |
|---|---|---|
| Dayan Mustafa (group head) | 23i-0702 | Phase 1 foundation (scaffolding, loader, baselines, metrics); `evidence_agent.py`, `decision_agent.py`, `agents/common.py` |
| Abdul Hadi | 23i-0707 | `text_agent.py`, `image_agent.py`, `orchestrator.py` |
| Muhammad Adeel | 23i-0739 | remaining work: calibration, run / merge / bootstrap / failure-analysis / robustness scripts, tests, results and documentation |

Each member commits their own work from their own GitHub account.

## 9. Limitations
- 150 evaluation posts: confidence intervals are wide and most differences between B2, the full system and the ablations are within noise.
- One run per configuration (temperature 0 plus caching); no repeated-run standard deviation.
- T2Agent numbers are published, on a different backbone and a different sample; not reproduced.
- The evidence agent sees only the caption, so it cannot detect image-side or mismatch errors; retrieval relevance was not measured.
- Free-tier API: val images and captions are sent to an external service; web search results change over time (hence the cache snapshot).
- The calibration set has only 60 posts.
