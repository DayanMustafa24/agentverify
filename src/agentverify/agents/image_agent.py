"""Image agent: looks for manipulation / AI generation and for caption-image mismatch."""
from ..contracts import PostInput
from .common import cached_call, to_verdict

SYSTEM = "You are an image forensics analyst for a fake-news detection team. Reply with one JSON object and nothing else."
USER = """Caption: {text}

Inspect the attached image. Decide:
- "image_false": the image looks AI-generated or edited (unnatural textures, warped text, inconsistent lighting)
- "mismatch": the image looks genuine but does not belong to this caption (different event, place or people)
- "real": the image looks genuine and fits the caption
You do not need to fact-check the caption itself.
Return JSON with keys:
"source": one of "real", "image_false", "mismatch"
"confidence": number 0-1, probability that your "source" answer is correct
"reason": one or two sentences
"""


def run(post: PostInput, client, cache_dir):
    parsed, cost = cached_call(cache_dir, "image", post.post_id, client, SYSTEM,
                               USER.format(text=post.text), post.image_path)
    return to_verdict("image", parsed, {"real", "image_false", "mismatch"}), cost
