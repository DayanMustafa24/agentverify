"""Paired bootstrap over posts for E3/E4/E5 comparisons. Reads results/agent/merged/eval_*.jsonl.
  python scripts/bootstrap_compare.py
Saves results/agent/bootstrap.csv. 'p_not_better' = share of resamples where A is not better than B."""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from agentverify.contracts import SOURCE_LABELS, RunRecord

D = Path("results/agent/merged")
B, SEED = 2000, 0
PAIRS = [("full", "b2"), ("full", "no_text"), ("full", "no_image"), ("full", "no_evidence"),
         ("full_cal", "full"), ("full", "b1"), ("b2", "b1")]
LOWER_BETTER = {"ece", "brier"}


def read(c):
    lines = (D / f"eval_{c}.jsonl").read_text(encoding="utf-8").splitlines()
    return {r.post.post_id: r for r in (RunRecord.model_validate_json(l) for l in lines if l.strip())}


def arrays(recs, ids):
    rs = [recs[i] for i in ids]
    gb = np.array([bool(r.post.gold_is_fake) for r in rs]); pb = np.array([r.final.is_fake for r in rs])
    g4 = np.array([r.post.gold_source.value for r in rs]); p4 = np.array([r.final.source_label.value for r in rs])
    conf = np.array([r.final.confidence for r in rs]); cor = (gb == pb).astype(float)
    return gb, pb, g4, p4, conf, cor


def ece(conf, cor, bins=15):
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(cor[idx == b].mean() - conf[idx == b].mean()) * (idx == b).sum() / len(conf)
               for b in range(bins) if (idx == b).any())


def metrics(a, s):
    gb, pb, g4, p4, conf, cor = (x[s] for x in a)
    return {"acc": cor.mean(), "bin_macroF1": f1_score(gb, pb, average="macro", zero_division=0),
            "4way_macroF1": f1_score(g4, p4, labels=SOURCE_LABELS, average="macro", zero_division=0),
            "ece": ece(conf, cor), "brier": ((conf - cor) ** 2).mean()}


data = {c: read(c) for c in {x for p in PAIRS for x in p}}
rng = np.random.default_rng(SEED)
rows = []
for A, Bn in PAIRS:
    ids = sorted(set(data[A]) & set(data[Bn]))
    aa, bb = arrays(data[A], ids), arrays(data[Bn], ids)
    full = np.arange(len(ids))
    obs = {k: metrics(aa, full)[k] - metrics(bb, full)[k] for k in metrics(aa, full)}
    boots = {k: [] for k in obs}
    for _ in range(B):
        s = rng.integers(0, len(ids), len(ids))
        ma, mb = metrics(aa, s), metrics(bb, s)
        for k in obs:
            boots[k].append(ma[k] - mb[k])
    for k in obs:
        d = np.array(boots[k]); lo, hi = np.percentile(d, [2.5, 97.5])
        worse = (d >= 0).mean() if k in LOWER_BETTER else (d <= 0).mean()
        rows.append({"A": A, "B": Bn, "n": len(ids), "metric": k, "diff_A_minus_B": round(obs[k], 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3), "p_not_better": round(worse, 3)})
        print(rows[-1])
with open("results/agent/bootstrap.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("saved results/agent/bootstrap.csv  (ece/brier here are computed by this script; compare with summary.csv)")
