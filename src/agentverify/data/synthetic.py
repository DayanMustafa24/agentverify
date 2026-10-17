"""Builds a tiny MMFakeBench-shaped folder with generated images.
Used only for tests and the smoke test. It has no scientific value."""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = [("Real", "Original"), ("Fake", "Textual Veracity Distortion"),
           ("Fake", "Visual Veracity Distortion"), ("Fake", "Cross-modal Consistency Distortion")]


def build_synthetic(root: str | Path, n_val: int = 24, n_test: int = 80, seed: int = 0) -> Path:
    root = Path(root)
    rng = random.Random(seed)
    for split, n in (("val", n_val), ("test", n_test)):
        base = root / f"MMFakeBench_{split}"
        (base / "source").mkdir(parents=True, exist_ok=True)
        items = []
        for i in range(n):
            gt, cls = CLASSES[i % 4]
            sub = "real" if gt == "Real" else "fake"
            (base / sub).mkdir(exist_ok=True)
            rel = f"{sub}/img_{i}.jpg"
            col = tuple(rng.randint(30, 220) for _ in range(3))
            im = Image.new("RGB", (96, 96), col)
            ImageDraw.Draw(im).rectangle([10, 10, 10 + (i * 7) % 60, 40], fill=(255, 255, 255))
            im.save(base / rel)
            items.append({"text": f"Synthetic claim number {i} about event {rng.randint(0, 999)}",
                          "image_path": rel, "text_source": "synthetic", "image_source": "synthetic",
                          "gt_answers": gt, "fake_cls": cls})
        (base / "source" / f"MMFakeBench_{split}.json").write_text(json.dumps(items))
    return root
