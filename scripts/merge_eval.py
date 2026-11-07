"""Merge shards, fit Platt calibration on the cal set, evaluate every config, save results/agent/summary.csv.
  python scripts/merge_eval.py --shards results/agent/shard0 results/agent/shard1 results/agent/shard2"""
import argparse
import csv
from pathlib import Path

from agentverify.calibration import Platt
from agentverify.contracts import RunRecord
from agentverify.eval.report import evaluate_file, write_records

ap = argparse.ArgumentParser()
ap.add_argument("--shards", nargs="+", required=True)
ap.add_argument("--out-dir", default="results/agent/merged")
args = ap.parse_args()


def read(p):
    path = Path(p)

    print(f"[merge] Reading: {path}")

    raw = path.read_bytes()

    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        print(f"[merge] UTF-8 failed, trying CP1252: {path}")
        content = raw.decode("cp1252")

    return [
        RunRecord.model_validate_json(line)
        for line in content.splitlines()
        if line.strip()
    ]

def collect(role, cfg):
    recs = []
    for s in args.shards:
        f = Path(s) / f"{role}_{cfg}.jsonl"
        if f.exists():
            recs += read(f)
    return recs


out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
cal = collect("cal", "full")
platt = Platt().fit([r.final.confidence for r in cal],
                    [int(r.final.is_fake == r.post.gold_is_fake) for r in cal])
print(f"calibration set: {len(cal)} posts, identity={platt.identity}")

sets = {c: collect("eval", c) for c in ["b2", "full", "no_text", "no_image", "no_evidence"]}
if sets["full"]:
    cp = []
    for r in sets["full"]:
        r2 = r.model_copy(deep=True)
        r2.method = "agent_full_cal"
        r2.final.confidence = float(platt.apply([r.final.confidence])[0])
        cp.append(r2)
    sets["full_cal"] = cp

rows = []
for c, recs in sets.items():
    if not recs:
        continue
    f = str(out / f"eval_{c}.jsonl")
    write_records(f, recs)
    m = evaluate_file(f)
    n = len(recs)
    rows.append({"config": c, "n": m["n"], "parse_fail": round(m["parse_fail_rate"], 3),
                 "bin_acc": round(m["binary"]["accuracy"], 3), "bin_macroF1": round(m["binary"]["macro_f1"], 3),
                 "4way_macroF1": round(m["four_way"]["macro_f1"], 3),
                 "ECE": round(m["calibration"]["ece_15bins"], 3), "Brier": round(m["calibration"]["brier"], 3),
                 "tokens/post": round(sum(r.cost.prompt_tokens + r.cost.completion_tokens for r in recs) / n),
                 "llm_calls/post": round(sum(r.cost.llm_calls for r in recs) / n, 2),
                 "search_calls/post": round(sum(r.cost.search_calls for r in recs) / n, 2),
                 "latency_s/post": round(sum(r.cost.latency_s for r in recs) / n, 1)})
    print(rows[-1])
with open(out / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("saved", out / "summary.csv")
