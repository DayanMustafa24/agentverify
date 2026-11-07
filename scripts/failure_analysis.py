"""Failure analysis of the full agent system (4-way label errors).
  python scripts/failure_analysis.py
Saves results/agent/failure_categories.csv and results/agent/failure_cases.md (2 successes + 2 failures with trajectories).
Uses logs/traces/eval_full_shard*.jsonl (copy teammates' trace files there) for the raw decision output."""
import csv
import json
from collections import Counter
from pathlib import Path

from agentverify.contracts import RunRecord

recs = [RunRecord.model_validate_json(l) for l in
        Path("results/agent/merged/eval_full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
recs.sort(key=lambda r: r.post.post_id)
traces = {}
for f in Path("logs/traces").glob("eval_full_shard*.jsonl"):
    for l in f.read_text(encoding="utf-8").splitlines():
        if l.strip():
            t = json.loads(l); traces[t["post_id"]] = t   # later lines override duplicates


def category(r):
    g, p = r.post.gold_source.value, r.final.source_label.value
    agent_right = any(v.flagged_source.value == g for v in r.verdicts)
    ev = next((v for v in r.verdicts if v.agent == "evidence"), None)
    no_ev = ev is not None and not ev.evidence
    return f"{g} -> {p}", ("decision overrode a correct specialist" if agent_right else "all specialists wrong"), no_ev


wrong = [r for r in recs if r.final.source_label != r.post.gold_source]
right = [r for r in recs if r.final.source_label == r.post.gold_source]
cats = Counter(); detail = Counter()
for r in wrong:
    pair, why, no_ev = category(r)
    cats[pair] += 1; detail[(pair, why)] += 1
print(f"4-way errors: {len(wrong)}/{len(recs)}")
with open("results/agent/failure_categories.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["gold -> predicted", "cause", "count", "share_of_errors"])
    for (pair, why), n in sorted(detail.items(), key=lambda x: -x[1]):
        w.writerow([pair, why, n, round(n / len(wrong), 3)]); print(pair, "|", why, "|", n)
no_ev_n = sum(1 for r in wrong if category(r)[2])
print(f"errors where the evidence agent retrieved nothing: {no_ev_n}/{len(wrong)}")


def show(r, title):
    t = traces.get(r.post.post_id, {})
    out = [f"### {title}: {r.post.post_id}", f"- Caption: {r.post.text[:300]}",
           f"- Gold: {r.post.gold_source.value} | Predicted: {r.final.source_label.value} (confidence {r.final.confidence:.2f})"]
    for v in r.verdicts:
        out.append(f"- {v.agent} agent: flags {v.flagged_source.value} ({v.confidence:.2f}). {v.rationale}")
        for e in v.evidence[:3]:
            out.append(f"    - evidence: {e.domain} | {e.title[:80]}")
    out.append(f"- Decision explanation: {r.final.explanation}")
    out.append(f"- Cost: {r.cost.llm_calls} LLM calls, {r.cost.prompt_tokens + r.cost.completion_tokens} tokens")
    return "\n".join(out) + "\n"


top = [p for p, _ in cats.most_common(2)]
fail_pick = [next(r for r in wrong if category(r)[0] == p) for p in top]
succ_pick = [next(r for r in right if r.post.gold_source.value == g) for g in ("real", "mismatch")
             if any(r.post.gold_source.value == g for r in right)][:2]
md = ["# Failure analysis (full system, eval set)\n", f"4-way errors: {len(wrong)}/{len(recs)}\n"]
md += [show(r, "Success") for r in succ_pick] + [show(r, "Failure") for r in fail_pick]
Path("results/agent/failure_cases.md").write_text("\n".join(md), encoding="utf-8")
print("saved results/agent/failure_categories.csv and failure_cases.md")
