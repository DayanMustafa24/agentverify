from agentverify.baselines.single_vlm import run_b2
from agentverify.data.mmfakebench import load_split
from agentverify.data.synthetic import build_synthetic
from agentverify.eval.report import evaluate_file, read_records, write_records
from agentverify.llm import MockClient


def test_b2_end_to_end_with_mock(tmp_path):
    root = build_synthetic(tmp_path / "mm", n_val=40, n_test=40)
    posts = load_split(root, "val")
    recs = [run_b2(p, MockClient()) for p in posts]
    path = tmp_path / "r.jsonl"
    write_records(path, recs)
    assert len(read_records(path)) == 40
    m = evaluate_file(path, tmp_path / "out", n_boot=50)
    assert m["n"] == 40 and m["parse_fail_rate"] == 0
    assert 0 <= m["calibration"]["ece_15bins"] <= 1
    assert (tmp_path / "out" / "r_metrics.json").exists()
    assert (tmp_path / "out" / "r_reliability.png").exists()


def test_b2_handles_model_garbage(tmp_path):
    root = build_synthetic(tmp_path / "mm", n_val=8, n_test=8)
    posts = load_split(root, "val")
    bad = MockClient(fn=lambda s, u: None)
    recs = [run_b2(p, bad) for p in posts]
    assert all(not r.final.parse_ok for r in recs)
    m = evaluate_file((write_records(tmp_path / "g.jsonl", recs), tmp_path / "g.jsonl")[1], n_boot=0)
    assert m["parse_fail_rate"] == 1.0
