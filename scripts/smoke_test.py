"""End-to-end offline check (no API key, no dataset, no internet).
Builds a tiny synthetic MMFakeBench-shaped folder, makes splits, runs B2 with the mock client
and B1 with the dummy extractor, and evaluates both. Numbers are meaningless; this only proves the pipeline runs.
  python scripts/smoke_test.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from agentverify.baselines.clip_lr import ClipLR, DummyExtractor, build_features
from agentverify.baselines.single_vlm import run_b2
from agentverify.contracts import FinalVerdict, RunRecord, SourceLabel
from agentverify.data.mmfakebench import load_split
from agentverify.data.splits import SIZES, make_splits
from agentverify.data.synthetic import build_synthetic
from agentverify.eval.report import evaluate_file, write_records
from agentverify.llm import MockClient

out = Path("results/smoke"); out.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    root = build_synthetic(Path(tmp) / "MMFakeBench", n_val=24, n_test=200)
    val, test = load_split(root, "val"), load_split(root, "test")
    sizes = make_splits(val, test, Path(tmp) / "splits", sizes={"train": 100, "cal": 30, "test": 40, "rob": 10})
    print("splits:", sizes)

    client = MockClient()
    recs = [run_b2(p, client) for p in val]
    write_records(out / "b2_mock_val.jsonl", recs)
    m2 = evaluate_file(out / "b2_mock_val.jsonl", n_boot=0)

    ext = DummyExtractor()
    tr = test[:100]
    model = ClipLR().fit(build_features(*ext.embed(tr)), tr)
    pb, pm, ml = model.predict(build_features(*ext.embed(val)))
    recs1 = []
    for p, a, b in zip(val, pb, pm):
        fake = bool(a[1] >= 0.5)
        src = SourceLabel(ml[int(np.argmax(b))]) if fake else SourceLabel.REAL
        if fake and src == SourceLabel.REAL:
            src = SourceLabel.TEXT_FALSE
        recs1.append(RunRecord(method="b1", post=p, final=FinalVerdict(
            post_id=p.post_id, is_fake=fake, source_label=src, confidence=float(max(a)))))
    write_records(out / "b1_dummy_val.jsonl", recs1)
    m1 = evaluate_file(out / "b1_dummy_val.jsonl", n_boot=0)

print("B2(mock)  binary acc %.2f  ECE %.2f" % (m2["binary"]["accuracy"], m2["calibration"]["ece_15bins"]))
print("B1(dummy) binary acc %.2f  ECE %.2f" % (m1["binary"]["accuracy"], m1["calibration"]["ece_15bins"]))
print("SMOKE TEST PASSED (numbers above are not research results)")
