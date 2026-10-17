from collections import Counter

import pytest

from agentverify.contracts import SourceLabel
from agentverify.data.mmfakebench import load_split, normalize_source
from agentverify.data.splits import SIZES, make_splits, stratified_disjoint
from agentverify.data.synthetic import build_synthetic


def test_normalize_source():
    assert normalize_source("Original", "Real") == SourceLabel.REAL
    assert normalize_source("Textual Veracity Distortion", "Fake") == SourceLabel.TEXT_FALSE
    assert normalize_source("Visual Veracity Distortion", "Fake") == SourceLabel.IMAGE_FALSE
    assert normalize_source("Cross-modal Consistency Distortion", "Fake") == SourceLabel.MISMATCH
    with pytest.raises(ValueError):
        normalize_source("weird", "Fake")


def test_load_and_split(tmp_path):
    root = build_synthetic(tmp_path / "mm", n_val=20, n_test=400)
    val, test = load_split(root, "val", check_images=True), load_split(root, "test", check_images=True)
    assert len(val) == 20 and len(test) == 400
    assert Counter(p.gold_source for p in val)[SourceLabel.REAL] == 5
    sizes = {"train": 100, "cal": 50, "test": 80, "rob": 20}
    s = stratified_disjoint(test, sizes, seed=1)
    all_ids = [i for v in s.values() for i in v]
    assert len(all_ids) == len(set(all_ids)), "splits must be disjoint"
    for name, n in sizes.items():
        assert abs(len(s[name]) - n) <= 4
    # reproducible
    assert s == stratified_disjoint(test, sizes, seed=1)
    out = make_splits(val, test, tmp_path / "splits", sizes=sizes)
    assert out["val"] == 20
