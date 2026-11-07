"""Read run records (jsonl), write metrics.json and a reliability diagram."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..contracts import RunRecord
from .metrics import compute_all, reliability_bins


def read_records(path: str | Path) -> list[RunRecord]:
    return [RunRecord.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_records(path: str | Path, records: list[RunRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def reliability_plot(records: list[RunRecord], out_png: str | Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conf = np.array([r.final.confidence for r in records])
    corr = np.array([r.post.gold_is_fake == r.final.is_fake for r in records], float)
    rows = [r for r in reliability_bins(conf, corr, 10) if r[2] > 0]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot([r[3] for r in rows], [r[4] for r in rows], "o-", label="model")
    for r in rows:
        ax.annotate(str(r[2]), (r[3], r[4]), fontsize=6, xytext=(2, 4), textcoords="offset points")
    ax.set_xlabel("stated confidence"); ax.set_ylabel("accuracy in bin"); ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def evaluate_file(pred_path: str | Path, out_dir: str | Path | None = None, n_boot: int = 1000) -> dict:
    recs = read_records(pred_path)
    metrics = compute_all(recs, n_boot=n_boot)
    out_dir = Path(out_dir or Path(pred_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pred_path).stem
    (out_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2))
    reliability_plot(recs, out_dir / f"{stem}_reliability.png", stem)
    return metrics
