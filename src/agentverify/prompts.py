"""Prompts. B2 is the 'single VLM, no agents' baseline: one call, no retrieval."""

B2_SYSTEM = "You are a careful fact-checker for news posts that combine a caption and an image. Reply with one JSON object and nothing else."

B2_USER = """Caption: {text}

Look at the attached image together with the caption and decide if this post is misinformation.
Possible cases:
- real: caption is accurate and the image really shows what the caption describes
- text_false: the caption states something factually false
- image_false: the image is AI-generated or edited
- mismatch: the image is genuine but does not belong to the caption (out of context)

Return JSON with these keys:
"is_fake": true or false
"source": one of "real", "text_false", "image_false", "mismatch"
"confidence": a number between 0 and 1, your probability that your is_fake answer is correct
"reason": one or two sentences
"""
