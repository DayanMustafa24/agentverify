"""E1: B1 (CLIP + LR) trained on D_train (test-derived), evaluated on the SAME 150 val posts used by B2 and the agents.
  python scripts/b1_on_eval_set.py --root data/raw/MMFakeBench --n 150
Writes results/agent/merged/eval_b1.jsonl (then add ('full','b1') and ('b2','b1') to PAIRS in bootstrap_compare.py)."""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import get_posts  # noqa: E402

from agentverify.baselines.clip_lr import ClipExtractor, ClipLR, build_features  # noqa: E402
from agentverify.contracts import FinalVerdict, RunRecord, SourceLabel  # noqa: E402
from agentverify.eval.report import evaluate_file, write_records  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--out", default="results/agent/merged/eval_b1.jsonl")
args = ap.parse_args()

# identical sampling code to scripts/run_agent.py (role=eval) so the 150 posts are the same
posts = sorted(get_posts(args.root, "val"), key=lambda p: p.post_id)
rng = random.Random(42)
groups = defaultdict(list)
for p in posts:
    groups[p.gold_source.value].append(p)
picked = []
for g in groups.values():
    picked += rng.sample(g, min(max(1, round(args.n * len(g) / len(posts))), len(g)))
picked.sort(key=lambda p: p.post_id)
ev = picked
tr = get_posts(args.root, "train")
assert not ({p.image_path for p in tr} & {p.image_path for p in ev}), "train/eval overlap"
print(f"train={len(tr)} (D_train) eval={len(ev)} (same posts as the agent runs)")

ext = ClipExtractor()
Xtr = build_features(*ext.embed(tr))
Xev = build_features(*ext.embed(ev))
model = ClipLR().fit(Xtr, tr)
pb, pm, mlabels = model.predict(Xev)
recs = []
for p, pbi, pmi in zip(ev, pb, pm):
    fake = bool(pbi[1] >= 0.5)
    src = SourceLabel(mlabels[int(np.argmax(pmi))])
    if not fake:
        src = SourceLabel.REAL
    elif src == SourceLabel.REAL:
        src = SourceLabel.TEXT_FALSE
    recs.append(RunRecord(method="b1_clip_lr", post=p, final=FinalVerdict(
        post_id=p.post_id, is_fake=fake, source_label=src, confidence=float(max(pbi)))))
write_records(args.out, recs)
m = evaluate_file(args.out)
print(f"n={m['n']} binary acc={m['binary']['accuracy']:.3f} macroF1={m['binary']['macro_f1']:.3f} | "
      f"4-way macroF1={m['four_way']['macro_f1']:.3f} | ECE={m['calibration']['ece_15bins']:.3f} "
      f"Brier={m['calibration']['brier']:.3f}")
