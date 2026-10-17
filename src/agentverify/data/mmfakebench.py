"""MMFakeBench loader.

Expected layout (as in the official GitHub/HF release):
  <root>/MMFakeBench_val/source/MMFakeBench_val.json   + real/ + fake/
  <root>/MMFakeBench_test/source/MMFakeBench_test.json + real/ + fake/
Fields per item: text, image_path, text_source, image_source, gt_answers ("Fake"/"Real"), fake_cls.
The raw fake_cls strings are normalised by keyword, run scripts/inspect_dataset.py to confirm.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..contracts import PostInput, SourceLabel


def normalize_source(fake_cls: Optional[str], gt: Optional[str]) -> SourceLabel:
    g = (gt or "").strip().lower()
    c = (fake_cls or "").strip().lower()
    if g == "real" or c in {"", "real", "original", "none"}:
        return SourceLabel.REAL
    if "mismatch" in c or "consist" in c or "cross" in c or "ccd" in c:
        return SourceLabel.MISMATCH
    if "visual" in c or "image" in c or "vvd" in c:
        return SourceLabel.IMAGE_FALSE
    if "text" in c or "tvd" in c:
        return SourceLabel.TEXT_FALSE
    raise ValueError(f"Unrecognised fake_cls value: {fake_cls!r}. Update normalize_source().")


def _json_path(root: Path, split: str) -> Path:
    return root / f"MMFakeBench_{split}" / "source" / f"MMFakeBench_{split}.json"


def _resolve_image(root: Path, split: str, rel: str) -> str:
    base = root / f"MMFakeBench_{split}"
    # json stores paths like "\real\bbc_val_50\BBC_val_0.png"; a leading slash makes
    # pathlib discard `base`, so strip it and use forward slashes.
    rel = rel.replace("\\", "/").lstrip("/")
    for cand in (base / rel, base / "source" / rel, root / rel):
        if cand.exists():
            return str(cand)
    return str(base / rel)  # may not exist; caller can check


def load_split(root: str | Path, split: str, check_images: bool = False) -> list[PostInput]:
    root = Path(root)
    p = _json_path(root, split)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. See README, section 'Get the data'.")
    raw = json.loads(p.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else next(v for v in raw.values() if isinstance(v, list))
    posts = []
    for i, it in enumerate(items):
        src = normalize_source(it.get("fake_cls"), it.get("gt_answers"))
        img = _resolve_image(root, split, it["image_path"])
        if check_images and not Path(img).exists():
            raise FileNotFoundError(img)
        posts.append(PostInput(
            post_id=f"{split}-{i}", text=str(it["text"]), image_path=img,
            gold_is_fake=(src != SourceLabel.REAL), gold_source=src))
    return posts
