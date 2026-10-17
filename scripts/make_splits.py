"""Create the fixed evaluation splits (ids only). usage: python scripts/make_splits.py --root data/raw/MMFakeBench"""
import argparse

from agentverify.data.mmfakebench import load_split
from agentverify.data.splits import make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="data/splits")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
print(make_splits(load_split(args.root, "val"), load_split(args.root, "test"), args.out, args.seed))
