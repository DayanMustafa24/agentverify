from agentverify.baselines.single_vlm import parse_reply
from agentverify.contracts import AgentVerdict, SourceLabel
from agentverify.llm import extract_json


def test_extract_json_with_fence_and_noise():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"is_fake": true, "confidence": 0.7} hope it helps') == {"is_fake": True, "confidence": 0.7}
    assert extract_json("no json here") is None


def test_parse_reply_variants():
    v = parse_reply("p1", {"is_fake": True, "source": "mismatch", "confidence": 0.9, "reason": "x"})
    assert v.is_fake and v.source_label == SourceLabel.MISMATCH and v.parse_ok
    v = parse_reply("p2", {"is_fake": "false", "source": "text_false", "confidence": "0.6"})
    assert (not v.is_fake) and v.source_label == SourceLabel.REAL
    v = parse_reply("p3", {"is_fake": True, "source": "real", "confidence": 0.5})
    assert v.source_label == SourceLabel.TEXT_FALSE        # fake can never map to real
    v = parse_reply("p4", None)
    assert not v.parse_ok and v.confidence == 0.5


def test_confidence_is_clipped():
    assert AgentVerdict(agent="text", confidence=1.7).confidence == 1.0
    assert AgentVerdict(agent="text", confidence=-3).confidence == 0.0
