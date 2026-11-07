"""Text agent: reads only the caption and looks for false or misleading claims."""
from ..contracts import PostInput
from .common import cached_call, to_verdict

SYSTEM = "You are a text analyst for a fake-news detection team. Reply with one JSON object and nothing else."
USER = """Caption: {text}

You see only the caption, not the image. Judge whether the claim is likely factually false or
misleading, using the language (sensationalism, absolute claims, implausible details) and what you know.
Return JSON with keys:
"source": "text_false" if the caption itself is likely false or misleading, otherwise "real"
"confidence": number 0-1, probability that your "source" answer is correct
"reason": one or two sentences
"""


def run(post: PostInput, client, cache_dir):
    parsed, cost = cached_call(cache_dir, "text", post.post_id, client, SYSTEM, USER.format(text=post.text))
    return to_verdict("text", parsed, {"real", "text_false"}), cost
