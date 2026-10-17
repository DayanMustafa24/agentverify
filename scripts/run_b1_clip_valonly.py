"""B1 (Phase 1, val-only): 70/30 stratified split of the official val set.
  python scripts/run_b1_clip_valonly.py --root data/raw/MMFakeBench --out results/b1_val30.jsonl
"""
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from agentverify.baselines.clip_lr import ClipExtractor, ClipLR, build_features
from agentverify.contracts import FinalVerdict, RunRecord, SourceLabel
from agentverify.data.mmfakebench import load_split
from agentverify.eval.report import evaluate_file, write_records

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

posts = load_split(args.root, "val")
tr, ev = train_test_split(posts, test_size=0.3, random_state=42,
                          stratify=[p.gold_source.value for p in posts])
print(f"train={len(tr)} eval={len(ev)}")

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
      f"4-way macroF1={m['four_way']['macro_f1']:.3f} | ECE={m['calibration']['ece_15bins']:.3f}")