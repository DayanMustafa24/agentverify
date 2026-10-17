"""B2: a single VLM call per post, no tools, no agents."""
from __future__ import annotations

from ..contracts import CostLog, FinalVerdict, PostInput, RunRecord, SourceLabel
from ..llm import VLMClient
from ..prompts import B2_SYSTEM, B2_USER


def _to_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "fake", "yes"}:
            return True
        if s in {"false", "real", "no"}:
            return False
    return None


def parse_reply(post_id: str, reply: dict | None) -> FinalVerdict:
    """Parse failures are kept visible: predicted 'real' with confidence 0.5 and parse_ok=False."""
    if not reply:
        return FinalVerdict(post_id=post_id, is_fake=False, source_label=SourceLabel.REAL,
                            confidence=0.5, parse_ok=False, explanation="parse failure")
    fake = _to_bool(reply.get("is_fake"))
    if fake is None:
        return FinalVerdict(post_id=post_id, is_fake=False, source_label=SourceLabel.REAL,
                            confidence=0.5, parse_ok=False, explanation="bad is_fake")
    try:
        src = SourceLabel(str(reply.get("source", "")).strip().lower())
    except ValueError:
        src = SourceLabel.TEXT_FALSE if fake else SourceLabel.REAL
    if fake and src == SourceLabel.REAL:
        src = SourceLabel.TEXT_FALSE
    if not fake:
        src = SourceLabel.REAL
    try:
        conf = float(reply.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return FinalVerdict(post_id=post_id, is_fake=fake, source_label=src, confidence=conf,
                        explanation=str(reply.get("reason", ""))[:400])


def run_b2(post: PostInput, client: VLMClient) -> RunRecord:
    reply, cost = client.chat_json(B2_SYSTEM, B2_USER.format(text=post.text), post.image_path)
    return RunRecord(method="b2_single_vlm", post=post, final=parse_reply(post.post_id, reply), cost=cost)
