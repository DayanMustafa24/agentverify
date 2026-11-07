"""Merge robustness shards. For each config: metrics per condition, delta vs clean, prediction flip rate.
  python scripts/merge_robust.py --shards results/robust/shard0 results/robust/shard1 results/robust/shard2"""
import argparse
import csv
from pathlib import Path

from agentverify.contracts import RunRecord
from agentverify.eval.report import evaluate_file, write_records
from agentverify.robustness import original_id

ap = argparse.ArgumentParser()
ap.add_argument("--shards", nargs="+", required=True)
ap.add_argument("--out-dir", default="results/robust/merged")
args = ap.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)


def read(p):
    return [RunRecord.model_validate_json(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


files = {}
for s in args.shards:
    for f in Path(s).glob("rob_*.jsonl"):
        cond, cfg = f.stem[len("rob_"):].rsplit("_", 1)
        files.setdefault((cfg, cond), []).extend(read(f))

rows = []
for cfg in sorted({c for c, _ in files}):
    clean = {original_id(r.post.post_id): r for r in files.get((cfg, "clean"), [])}
    for cond in sorted(k[1] for k in files if k[0] == cfg):
        recs = files[(cfg, cond)]
        path = str(out / f"rob_{cond}_{cfg}.jsonl")
        write_records(path, recs)
        m = evaluate_file(path)
        paired = [(clean[original_id(r.post.post_id)], r) for r in recs if original_id(r.post.post_id) in clean]
        flip = sum(a.final.is_fake != b.final.is_fake for a, b in paired) / max(1, len(paired))
        flip4 = sum(a.final.source_label != b.final.source_label for a, b in paired) / max(1, len(paired))
        rows.append({"config": cfg, "condition": cond, "n": m["n"], "bin_macroF1": round(m["binary"]["macro_f1"], 3),
                     "4way_macroF1": round(m["four_way"]["macro_f1"], 3),
                     "ECE": round(m["calibration"]["ece_15bins"], 3), "parse_fail": round(m["parse_fail_rate"], 3),
                     "binary_flip_vs_clean": round(flip, 3), "4way_flip_vs_clean": round(flip4, 3)})
base = {(r["config"]): r for r in rows if r["condition"] == "clean"}
for r in rows:
    b = base.get(r["config"])
    r["d_bin_F1"] = round(r["bin_macroF1"] - b["bin_macroF1"], 3) if b else ""
    r["d_4way_F1"] = round(r["4way_macroF1"] - b["4way_macroF1"], 3) if b else ""
    r["d_ECE"] = round(r["ECE"] - b["ECE"], 3) if b else ""
    print(r)
with open(out / "robustness_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("saved", out / "robustness_summary.csv")
