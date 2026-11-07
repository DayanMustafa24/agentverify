import pytest

from agentverify.agents.common import QuotaGuard, RateLimited, cached_call, to_verdict
from agentverify.agents.orchestrator import run_pipeline
from agentverify.calibration import Platt
from agentverify.contracts import CostLog, PostInput, SourceLabel
from agentverify.llm import MockClient
from agentverify.tools.search import EvidenceTool


def _tool(tmp_path):
    return EvidenceTool(backends={
        "wikipedia": lambda q, k: [{"title": "T", "url": "https://en.wikipedia.org/wiki/T", "snippet": "s"}],
        "web": lambda q, k: []}, cache_dir=tmp_path / "search")


def _post(i=0):
    return PostInput(post_id=f"p{i}", text=f"claim number {i}", image_path="x.png",
                     gold_is_fake=bool(i % 2),
                     gold_source=SourceLabel.TEXT_FALSE if i % 2 else SourceLabel.REAL)


class FlakyClient:
    def __init__(self):
        self.calls = 0

    def chat_json(self, system, user_text, image_path=None):
        self.calls += 1
        if self.calls == 1:
            return None, CostLog(llm_calls=1)
        return {"x": 1}, CostLog(llm_calls=1)


def test_cached_call_does_not_cache_failures(tmp_path):
    c = FlakyClient()
    assert cached_call(tmp_path, "a", "p1", c, "s", "u")[0] is None
    assert cached_call(tmp_path, "a", "p1", c, "s", "u")[0] == {"x": 1}
    assert cached_call(tmp_path, "a", "p1", c, "s", "u")[0] == {"x": 1}   # served from cache
    assert c.calls == 2


def test_to_verdict_clips_restricts_and_flags_failure():
    v = to_verdict("text", {"source": "image_false", "confidence": 7, "reason": "r"}, {"real", "text_false"})
    assert v.flagged_source == SourceLabel.REAL and v.confidence == 1.0
    assert to_verdict("text", {"source": "text_false", "confidence": "abc"}, {"real", "text_false"}).confidence == 0.5
    assert "AGENT FAILED" in to_verdict("image", None, {"real"}).rationale


def test_platt_identity_on_tiny_or_single_class_data():
    assert Platt().fit([0.9] * 5, [1] * 5).identity
    assert Platt().fit([0.9] * 20, [1] * 20).identity


def test_platt_is_monotone():
    conf = [0.6] * 20 + [0.9] * 20
    correct = [1] * 10 + [0] * 10 + [1] * 18 + [0] * 2
    p = Platt().fit(conf, correct)
    assert not p.identity
    lo, hi = p.apply([0.6])[0], p.apply([0.9])[0]
    assert 0 <= lo < hi <= 1


def test_rate_limiter_enforces_call_budget():
    rl = RateLimited(MockClient(), rpm=60000, max_calls=2)
    rl.chat_json("s", "claim a")
    rl.chat_json("s", "claim b")
    with pytest.raises(QuotaGuard):
        rl.chat_json("s", "claim c")


@pytest.mark.parametrize("flags,calls,n_verdicts", [
    ({}, 5, 3), ({"use_text": False}, 4, 2), ({"use_image": False}, 4, 2), ({"use_evidence": False}, 3, 2)])
def test_ablation_switches(tmp_path, flags, calls, n_verdicts):
    r = run_pipeline(_post(1), MockClient(), _tool(tmp_path), tmp_path / "cache",
                     trace_path=tmp_path / "t.jsonl", **flags)
    assert len(r.verdicts) == n_verdicts
    assert r.cost.llm_calls == calls
    assert (tmp_path / "t.jsonl").read_text().strip()


def test_decision_failure_falls_back_to_specialist(tmp_path):
    class NoDecision(MockClient):
        def chat_json(self, system, user_text, image_path=None):
            if "decision agent" in system.lower():
                return None, CostLog(llm_calls=1)
            return super().chat_json(system, user_text, image_path)

    r = run_pipeline(_post(1), NoDecision(), _tool(tmp_path), tmp_path / "cache")
    assert r.final.parse_ok is False
    assert r.final.explanation.startswith("fallback")
