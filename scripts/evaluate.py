"""Recompute metrics + reliability diagram for a records file.  python scripts/evaluate.py results/b2_val.jsonl"""
import json
import sys

from agentverify.eval.report import evaluate_file

print(json.dumps(evaluate_file(sys.argv[1]), indent=2))
