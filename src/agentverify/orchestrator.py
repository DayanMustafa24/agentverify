"""Runs the enabled agents for one post, handles failures, logs a trace, returns a RunRecord.
Ablation switches: use_text / use_image / use_evidence."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from ..baselines.single_vlm import parse_reply
from ..contracts import CostLog, PostInput, RunRecord, SourceLabel
from . import decision_agent, evidence_agent, image_agent, text_agent
from .common import QuotaGuard, failed_verdict

_trace_lock = threading.Lock()


def run_pipeline(post: PostInput, client, tool, cache_dir, use_text=True, use_image=True,
                 use_evidence=True, method="agent_full", trace_path=None) -> RunRecord:
    cost, verdicts = CostLog(), []
    for name, on in (("text", use_text), ("image", use_image), ("evidence", use_evidence)):
        if not on:
            continue
        try:
            if name == "text":
                v, c = text_agent.run(post, client, cache_dir)
            elif name == "image":
                v, c = image_agent.run(post, client, cache_dir)
            else:
                v, c = evidence_agent.run(post, client, cache_dir, tool)
        except QuotaGuard:
            raise
        except Exception as e:  # isolate one agent's crash from the pipeline
            v, c = failed_verdict(name, f"{type(e).__name__}: {e}"), CostLog()
        verdicts.append(v)
        cost.add(c)
    reply, c = decision_agent.run(post, verdicts, client, cache_dir)
    cost.add(c)
    final = parse_reply(post.post_id, reply)
    if not final.parse_ok and verdicts:  # recovery: fall back to the most confident specialist
        best = max(verdicts, key=lambda v: v.confidence)
        fake = best.flagged_source != SourceLabel.REAL
        final = final.model_copy(update={"is_fake": fake, "source_label": best.flagged_source,
                                         "confidence": best.confidence,
                                         "explanation": f"fallback to {best.agent} agent (decision parse failed)"})
    rec = RunRecord(method=method, post=post, verdicts=verdicts, final=final, cost=cost)
    if trace_path:
        row = {"method": method, "post_id": post.post_id, "caption": post.text,
               "gold": post.gold_source.value if post.gold_source else None,
               "agent_reports": [v.model_dump(mode="json") for v in verdicts],
               "decision_raw": reply, "final": final.model_dump(mode="json"),
               "cost": cost.model_dump()}
        with _trace_lock:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    return rec
