# Development log
Fill every [DATE] from `git log --format="%ad %s" --date=short` or your terminal history. Do not guess dates.

## 1. B2 first attempt returned only parse failures  [DATE]
- **Failure observed:** B2 on 60 val posts: `parse_fail_rate = 1.000`; metrics only reflected the "real, 0.5" fallback (accuracy 0.300, macro-F1 0.231).
- **Change:** `llm.py` now prints `[llm error] <type>: <message>`; image-path resolution for `\real\...` paths in the MMFakeBench json fixed in the loader (all 1000 images had shown as missing).
- **Measured effect:** valid B2 run with `parse_fail_rate = 0.000`; the invalid run is kept in `results/RESULTS.md` and not reported as a result.

## 2. Free-tier quota blocked the planned model  [DATE]
- **Failure observed:** AI Studio showed Gemini 2.5 Flash at 5 RPM / 20 requests per day; an agent pipeline needs about 9 calls per post.
- **Change:** switched all methods to `gemini-3.5-flash-lite` (15 RPM, 500 requests/day); added a per-call disk cache (`cached_call`), a rate limiter with a call budget (`RateLimited`, `--max-calls`) and 3-way sharding across members' own keys; ablations reuse the cached specialist reports and only repeat the decision call.
- **Measured effect:** full evaluation (150 posts x B2 + full + 3 ablations) completed within the daily quotas; B2 re-run on the same posts with the same model.

## 3. Evidence agent silently failed on every post  [DATE]
- **Failure observed:** first agent run: `search_calls/post = 0.0`, `llm_calls/post = 3.0` for the full system, and `no_evidence` identical to `full`. Trace showed `AGENT FAILED: TypeError: cannot unpack non-iterable NoneType`.
- **Cause:** `EvidenceTool.search` returned `None` after a bad edit; Wikipedia also returned `HTTPError` (missing contact in the User-Agent / rate limit), and failed searches had been cached as empty results.
- **Change:** `search()` always returns `(items, cost)` and no longer caches failures; `wikipedia_backend` sends a descriptive User-Agent and retries on 429; DuckDuckGo "no results" is non-fatal.
- **Measured effect (shard 0 preview, n=50):** `llm_calls/post` 3.0 -> 4.94, `search_calls/post` 0 -> 1.84; on all 150 posts (final): full 4.97 calls and 1.76 searches per post, full vs no_evidence 4-way macro-F1 0.672 vs 0.656.

## 4. jsonl write crashed on Windows  [DATE]
- **Failure observed:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2011'` in `write_records` once evidence snippets contained non-ASCII text.
- **Change:** `PYTHONUTF8=1` and `encoding="utf-8"` in `eval/report.py` read/write calls. All LLM replies were already cached, so no API calls were lost.
- **Measured effect:** all shards write and merge cleanly.

## 5. Calibration layer did not help  [DATE]
- **Observation:** Platt scaling fit on the 60-post calibration set: ECE 0.102 -> 0.106, Brier 0.133 -> 0.132 on the 150 eval posts (diff +0.004, 95% CI [-0.057, 0.064]).
- **Change / decision:** reported as a negative result for Gap 1; no tuning on the evaluation set (to avoid leakage). Possible causes: tiny calibration set, distribution shift between test-split and val posts.

## 6. B1 protocol corrected  [DATE]
- **Issue (TA feedback):** Phase 1 B1 trained on 700 val posts and evaluated on 300 val posts, overlapping D_val.
- **Change:** `scripts/b1_on_eval_set.py` trains on D_train (test-derived) and evaluates on the same 150 val posts as B2 and the agents; the 700/300 run is marked "Phase 1 preliminary, superseded".

## 7. Robustness suite made label-safe  [DATE]
- **Issue (TA feedback):** a crop can remove the evidence that makes a post fake.
- **Change:** `robustness.make_condition_post` applies crop to real posts only; jpeg / blur / typos / hashtags are label-preserving by construction; perturbed posts get a new post_id so they do not reuse clean caches. A hand check of K examples per condition is recorded in `docs/perturbation_check.md`.


## 8. Caption typo robustness

**Observation:** On the 60-post typos3 robustness sample, the full system changed its binary verdict on 20% of posts, compared with 5% for B2. The change in binary macro-F1 was -0.073 for the full system and +0.049 for B2. These results are suggestive, based on one run with n=60.

**Hypothesis (not tested):** The evidence agent's search query may change when the caption contains typos, potentially affecting the retrieved evidence and final verdict.

**Decision:** We did not run the `no_evidence` ablation on typos3 because of time and API quota constraints. This remains a limitation and the proposed explanation has not been experimentally verified.