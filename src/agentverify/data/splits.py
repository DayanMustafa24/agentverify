"""Reproducible splits (ids only, safe to commit; no MMFakeBench content is stored).

D_val   : official validation set (1,000), main comparison with T2Agent / MIRAGE
D_train : 2,000 from test, only used to fit B1 (CLIP + LR)
D_cal   : 1,000 from test, only used to fit the Phase 2 calibrator
D_test  : 2,000 from test, final held-out evaluation
D_rob   : 300 from test, perturbation experiments
All test-derived subsets are disjoint and stratified by 4-way label.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..contracts import PostInput

SIZES = {"train": 2000, "cal": 1000, "test": 2000, "rob": 300}


def stratified_disjoint(posts: list[PostInput], sizes: dict[str, int], seed: int = 42) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for p in posts:
        by_label[p.gold_source.value].append(p.post_id)
    for ids in by_label.values():
        rng.shuffle(ids)
    total = len(posts)
    out: dict[str, list[str]] = {k: [] for k in sizes}
    cursor = {lab: 0 for lab in by_label}
    for name, n in sizes.items():
        for lab, ids in by_label.items():
            take = int(round(n * len(ids) / total))
            out[name] += ids[cursor[lab]: cursor[lab] + take]
            cursor[lab] += take
    return out


def make_splits(val_posts, test_posts, out_dir: str | Path, seed: int = 42, sizes: dict | None = None) -> dict:
    sizes = sizes or SIZES
    if sum(sizes.values()) > len(test_posts):
        raise ValueError(f"Requested {sum(sizes.values())} test-derived posts but only {len(test_posts)} exist.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"val": [p.post_id for p in val_posts]}
    splits.update(stratified_disjoint(test_posts, sizes, seed))
    for name, ids in splits.items():
        (out_dir / f"{name}.json").write_text(json.dumps(ids))
    return {k: len(v) for k, v in splits.items()}


def load_ids(out_dir: str | Path, name: str) -> list[str]:
    return json.loads((Path(out_dir) / f"{name}.json").read_text())
