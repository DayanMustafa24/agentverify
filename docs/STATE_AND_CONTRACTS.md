# State and contracts
Per post the orchestrator keeps one `RunRecord`: input `PostInput`, a list of `AgentVerdict`
(agent name, flagged source, confidence in [0,1], rationale, evidence list), a `FinalVerdict`
(is_fake, 4-way source label, confidence, explanation, abstain flag, parse_ok) and a `CostLog`
(tokens, latency, LLM calls, search calls). Every baseline and every agent configuration
writes the same record, so one evaluator covers all comparisons and ablations.

## Phase 2 agents (`src/agentverify/agents/`)
| Agent | Input | Output (`AgentVerdict`) | Calls |
|---|---|---|---|
| text | caption only | flags `text_false` or `real`, confidence, rationale | 1 LLM |
| image | image + caption | flags `image_false`, `mismatch` or `real` | 1 vision LLM |
| evidence | caption | writes a search query, retrieves Wikipedia + web results (`tools/search.py`), judges supports / contradicts / insufficient; flags `text_false` only when contradicted; keeps the retrieved `EvidenceItem` list | 2 LLM + 2 searches |
| decision | caption + the enabled agents' reports as text | `FinalVerdict` parsed by the same parser as B2 | 1 LLM |

`orchestrator.run_pipeline(post, client, tool, cache_dir, use_text, use_image, use_evidence, method, trace_path)` runs the enabled agents,
catches a crashing agent and records it as a failed verdict (`AGENT FAILED: ...`), calls the decision agent, and if the decision reply cannot be parsed
falls back to the most confident specialist (`parse_ok=False`, explanation starts with `fallback`). One JSON line per post is appended to `trace_path`
(agent reports with retrieved evidence, raw decision reply, final verdict, cost).

Ablation switches: `use_text`, `use_image`, `use_evidence` (E5). Calibration (`calibration.py`, Platt scaling) is fit on the calibration set only and applied
after the run (`merge_eval.py`), so "no calibration" costs no extra calls.

## Caching and rate limiting (`agents/common.py`)
`cached_call` stores each successful LLM reply (and its cost) on disk keyed by agent, post id, prompts and model; failed calls are never cached.
`RateLimited` spaces calls to the requests-per-minute limit and stops a run at `--max-calls`. Perturbed posts use a new `post_id` (`<id>__<condition>`)
so they never read the clean post's cache.
