# State and contracts
Per post the orchestrator keeps one `RunRecord`: input `PostInput`, a list of `AgentVerdict`
(agent name, flagged source, confidence in [0,1], rationale, evidence list), a `FinalVerdict`
(is_fake, 4-way source label, calibrated confidence, explanation, abstain flag) and a `CostLog`
(tokens, latency, LLM calls, search calls). Every baseline and, later, every agent configuration
writes the same record, so one evaluator covers all comparisons and ablations.
