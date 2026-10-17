"""B1: CLIP features + logistic regression. Trains on split 'train', evaluates on 'val' (and optionally 'test').
  python scripts/run_b1_clip.py --root data/raw/MMFakeBench --out results/b1_val.jsonl
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import get_posts  # noqa: E402

from agentverify.baselines.clip_lr import ClipLR, ClipExtractor, DummyExtractor, build_features  # noqa: E402
from agentverify.contracts import FinalVerdict, RunRecord, SourceLabel  # noqa: E402
from agentverify.eval.report import evaluate_file, write_records  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--train-split", default="train")
ap.add_argument("--eval-split", default="val")
ap.add_argument("--extractor", choices=["clip", "dummy"], default="clip")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()

ext = ClipExtractor() if args.extractor == "clip" else DummyExtractor()
tr = get_posts(args.root, args.train_split, limit=args.limit)
ev = get_posts(args.root, args.eval_split, limit=args.limit)
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
      f"4-way macroF1={m['four_way']['macro_f1']:.3f} | ECE={m['calibration']['ece_15bins']:.3f}")
