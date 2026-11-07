"""Robustness conditions (Gap 2): build a perturbed copy of a post with a NEW post_id.
The new id matters: agent and B2 caches are keyed by post_id, so a perturbed post must never reuse the clean id.

Label-safety rules (TA feedback):
  - crop can remove the evidence of a manipulation, so it is applied to REAL posts only (returns None otherwise);
  - jpeg / blur / noise on images and typos / shout / hashtags on captions keep the label by construction
    (an AI-generated image stays AI-generated; a false caption stays false)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image

from . import perturb
from .contracts import PostInput, SourceLabel

# condition name -> (kind, perturbation, severity)
SPECS = {
    "jpeg3": ("image", "jpeg", 3),
    "blur2": ("image", "blur", 2),
    "crop2": ("image", "crop", 2),     # real posts only
    "typos3": ("text", "typos", 3),
    "hashtags2": ("text", "hashtags", 2),
}


def make_condition_post(post: PostInput, cond: str, out_dir: str | Path) -> Optional[PostInput]:
    if cond == "clean":
        return post
    kind, name, sev = SPECS[cond]
    if name == "crop" and post.gold_source != SourceLabel.REAL:
        return None
    pid = f"{post.post_id}__{cond}"
    if kind == "image":
        out = Path(out_dir) / f"{hashlib.md5(pid.encode()).hexdigest()}.png"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            perturb.IMAGE_PERTURBATIONS[name](Image.open(post.image_path), sev).save(out)
        return post.model_copy(update={"post_id": pid, "image_path": str(out)})
    seed = int(hashlib.md5(post.post_id.encode()).hexdigest(), 16) % (2 ** 31)
    return post.model_copy(update={"post_id": pid,
                                   "text": perturb.TEXT_PERTURBATIONS[name](post.text, sev, seed)})


def original_id(post_id: str) -> str:
    return post_id.split("__")[0]
