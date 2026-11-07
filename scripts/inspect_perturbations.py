"""Hand-check helper for docs/perturbation_check.md.
  python scripts/inspect_perturbations.py --root data/raw/MMFakeBench --per-cond 10
Uses the SAME D_rob sample as run_robustness.py (n=60, seed 42), rebuilds each perturbed copy with make_condition_post,
and writes to data/cache/perturb_check_sbs/ (git-ignored; contains dataset images, never commit):
  <cond>/NN_<class>.png   original | perturbed side by side   (image conditions)
  <cond>/NN_<class>.txt   original and perturbed caption
  hand_check.csv          one row per example; YOU fill label_preserved (Y/N) and note after looking
  automated_report.json   automatic sanity checks (these do NOT prove the label is preserved)"""
import argparse
import csv
import difflib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

sys.path.insert(0, str(Path(__file__).parent))
from _common import get_posts  # noqa: E402

from agentverify.robustness import SPECS, make_condition_post  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--n", type=int, default=60, help="same D_rob sample size as run_robustness.py")
ap.add_argument("--per-cond", type=int, default=10)
ap.add_argument("--conditions", nargs="+", default=["jpeg3", "typos3", "crop2"])
ap.add_argument("--out", default="data/cache/perturb_check_sbs")
args = ap.parse_args()

# identical sampling code to scripts/run_robustness.py
posts = sorted(get_posts(args.root, "rob"), key=lambda p: p.post_id)
rng = random.Random(42)
groups = defaultdict(list)
for p in posts:
    groups[p.gold_source.value].append(p)
picked = []
for g in groups.values():
    picked += rng.sample(g, min(max(1, round(args.n * len(g) / len(posts))), len(g)))
picked.sort(key=lambda p: p.post_id)

# round-robin over the 4 classes so every class is looked at
by_cls = defaultdict(list)
for p in picked:
    by_cls[p.gold_source.value].append(p)
order = []
while any(by_cls.values()):
    for c in sorted(by_cls):
        if by_cls[c]:
            order.append(by_cls[c].pop(0))

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
report, rows = {}, []
for cond in args.conditions:
    kind, name, sev = SPECS[cond]
    d = out / cond
    d.mkdir(exist_ok=True)
    stats = {"examined": 0, "skipped_by_rule": 0, "unchanged_image": 0, "unchanged_text": 0,
             "size_changed": 0, "label_fields_changed": 0, "mean_abs_pixel_diff": []}
    k = 0
    for post in order:
        if k >= args.per_cond:
            break
        pp = make_condition_post(post, cond, "data/cache/perturbed")
        if pp is None:                       # crop on a non-real post is refused on purpose
            stats["skipped_by_rule"] += 1
            continue
        k += 1
        stats["examined"] += 1
        if (pp.gold_source, pp.gold_is_fake) != (post.gold_source, post.gold_is_fake):
            stats["label_fields_changed"] += 1
        tag = f"{k:02d}_{post.gold_source.value}"
        a = Image.open(post.image_path).convert("RGB")
        b = Image.open(pp.image_path).convert("RGB")
        if kind == "image":
            if a.size != b.size:
                stats["size_changed"] += 1
            diff = ImageChops.difference(a, b.resize(a.size)) if a.size != b.size else ImageChops.difference(a, b)
            m = sum(ImageStat.Stat(diff).mean) / 3
            stats["mean_abs_pixel_diff"].append(round(m, 2))
            if m == 0:
                stats["unchanged_image"] += 1
            h = 360
            a2 = a.resize((int(a.width * h / a.height), h))
            b2 = b.resize((int(b.width * h / b.height), h))
            canvas = Image.new("RGB", (a2.width + b2.width + 10, h), "white")
            canvas.paste(a2, (0, 0))
            canvas.paste(b2, (a2.width + 10, 0))
            canvas.save(d / f"{tag}.png")
        else:
            if pp.text == post.text:
                stats["unchanged_text"] += 1
        ratio = difflib.SequenceMatcher(None, post.text, pp.text).ratio()
        (d / f"{tag}.txt").write_text(f"GOLD: {post.gold_source.value}\nORIGINAL: {post.text}\nPERTURBED: {pp.text}\n"
                                      f"text similarity: {ratio:.2f}\n", encoding="utf-8")
        rows.append({"condition": cond, "file": tag, "post_id": post.post_id, "gold": post.gold_source.value,
                     "label_preserved (Y/N)": "", "note": ""})
    md = stats.pop("mean_abs_pixel_diff")
    stats["mean_abs_pixel_diff_avg"] = round(sum(md) / len(md), 2) if md else None
    report[cond] = stats
    print(cond, stats)

(out / "automated_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
with open(out / "hand_check.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nOpen the images/captions in {out}, fill label_preserved in hand_check.csv, then write docs/perturbation_check.md.")
