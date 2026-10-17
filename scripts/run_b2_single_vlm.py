"""B2: single VLM, no agents.  usage:
  python scripts/run_b2_single_vlm.py --root data/raw/MMFakeBench --split val --limit 100 --out results/b2_val.jsonl
"""
import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _common import get_posts  # noqa: E402

from agentverify.baselines.single_vlm import run_b2  # noqa: E402
from agentverify.eval.report import evaluate_file, write_records  # noqa: E402
from agentverify.llm import MockClient, OpenAICompatClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--provider", choices=["openai", "mock"], default="openai")
ap.add_argument("--out", required=True)
args = ap.parse_args()

posts = get_posts(args.root, args.split, limit=args.limit)
client = OpenAICompatClient() if args.provider == "openai" else MockClient()
recs = [run_b2(p, client) for p in tqdm(posts)]
write_records(args.out, recs)
m = evaluate_file(args.out)
print(f"n={m['n']} parse_fail={m['parse_fail_rate']:.3f}")
print(f"binary acc={m['binary']['accuracy']:.3f} macroF1={m['binary']['macro_f1']:.3f} | "
      f"4-way macroF1={m['four_way']['macro_f1']:.3f} | ECE={m['calibration']['ece_15bins']:.3f} "
      f"Brier={m['calibration']['brier']:.3f}")
