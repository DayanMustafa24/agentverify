"""Shared helpers for agents: disk cache, rate limiting, verdict parsing."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from ..contracts import AgentVerdict, CostLog, SourceLabel


class QuotaGuard(RuntimeError):
    """Raised when the per-run API call budget is used up (rerun later; the cache resumes)."""


class RateLimited:
    """Wraps a VLM client: spaces calls to at most `rpm` per minute and caps total calls."""

    def __init__(self, inner, rpm: float = 12, max_calls: int | None = None):
        self.inner, self.interval, self.max_calls = inner, 60.0 / rpm, max_calls
        self.lock, self.next, self.calls = threading.Lock(), 0.0, 0

    def chat_json(self, system, user_text, image_path=None):
        with self.lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise QuotaGuard(f"call budget {self.max_calls} reached")
            self.calls += 1
            now = time.time()
            slot = max(now, self.next)
            self.next = slot + self.interval
        time.sleep(max(0.0, slot - now))
        return self.inner.chat_json(system, user_text, image_path)


def cached_call(cache_dir, agent, post_id, client, system, user, image_path=None, version="v1"):
    """One LLM call, cached on disk. Failed calls (None) are never cached."""
    key = hashlib.md5(f"{agent}|{post_id}|{version}|{system}|{user}".encode()).hexdigest()
    p = Path(cache_dir) / agent / f"{key}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return d["parsed"], CostLog(**d["cost"])
    parsed, cost = client.chat_json(system, user, image_path)
    if parsed is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"parsed": parsed, "cost": cost.model_dump()}), encoding="utf-8")
    return parsed, cost


def failed_verdict(agent: str, why: str) -> AgentVerdict:
    return AgentVerdict(agent=agent, flagged_source=SourceLabel.REAL, confidence=0.5,
                        rationale=f"AGENT FAILED: {why}"[:300])


def to_verdict(agent: str, parsed: dict | None, allowed: set[str], evidence=None) -> AgentVerdict:
    if not parsed:
        return failed_verdict(agent, "unparseable reply")
    try:
        src = SourceLabel(str(parsed.get("source", "real")).strip().lower())
    except ValueError:
        src = SourceLabel.REAL
    if src.value not in allowed:
        src = SourceLabel.REAL
    try:
        conf = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return AgentVerdict(agent=agent, flagged_source=src, confidence=conf,
                        rationale=str(parsed.get("reason", ""))[:300], evidence=evidence or [])
