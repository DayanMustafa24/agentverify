"""Sanity-check the downloaded MMFakeBench before running anything.
usage: python scripts/inspect_dataset.py --root data/raw/MMFakeBench
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from agentverify.data.mmfakebench import _json_path, load_split

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
args = ap.parse_args()

for split in ("val", "test"):
    p = _json_path(Path(args.root), split)
    if not p.exists():
        print(f"[{split}] MISSING {p}")
        continue
    raw = json.loads(p.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else next(v for v in raw.values() if isinstance(v, list))
    print(f"[{split}] {len(items)} items. keys: {sorted(items[0].keys())}")
    print("  raw gt_answers :", dict(Counter(i.get("gt_answers") for i in items)))
    print("  raw fake_cls   :", dict(Counter(i.get("fake_cls") for i in items)))
    posts = load_split(args.root, split)
    print("  4-way labels   :", dict(Counter(p.gold_source.value for p in posts)))
    missing = sum(1 for p in posts if not Path(p.image_path).exists())
    print(f"  missing images : {missing}/{len(posts)}")
    print("  sample         :", posts[0].text[:80], "|", posts[0].image_path)
