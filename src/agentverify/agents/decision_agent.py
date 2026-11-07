"""Decision agent: fuses the specialist reports (only the enabled ones) into the final verdict."""
from ..contracts import PostInput
from .common import cached_call

SYSTEM = "You are the decision agent of a fake-news detection team. Reply with one JSON object and nothing else."
USER = """Caption: {text}

Specialist reports (some specialists may be absent):
{reports}

The text agent only reads the caption. The image agent checks the image and caption-image consistency.
The evidence agent checks the caption's facts against retrieved sources. Weigh the reports; do not trust any
single agent blindly, and treat "insufficient" or failed reports as weak signals.
Cases: real, text_false (caption false), image_false (image AI-generated or edited), mismatch (genuine image, wrong caption).
Return JSON with keys:
"is_fake": true or false
"source": one of "real", "text_false", "image_false", "mismatch"
"confidence": number 0-1, probability that your is_fake answer is correct
"reason": one or two sentences explaining the verdict"""


def format_reports(verdicts) -> str:
    if not verdicts:
        return "(none)"
    return "\n".join(f"- {v.agent} agent: flags={v.flagged_source.value}, confidence={v.confidence:.2f}. {v.rationale}"
                     for v in verdicts)


def run(post: PostInput, verdicts, client, cache_dir):
    return cached_call(cache_dir, "decision", post.post_id, client, SYSTEM,
                       USER.format(text=post.text, reports=format_reports(verdicts)))
