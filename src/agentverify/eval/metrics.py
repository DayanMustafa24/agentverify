"""Classification, calibration and selective-prediction metrics (numpy only)."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..contracts import SOURCE_LABELS, RunRecord


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def per_class_recall(y_true, y_pred, labels) -> dict:
    out = {}
    for lab in labels:
        idx = [i for i, t in enumerate(y_true) if t == lab]
        out[str(lab)] = (sum(1 for i in idx if y_pred[i] == lab) / len(idx)) if idx else float("nan")
    return out


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins on [0,1]."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    """Brier score of 'my answer is correct' predictions."""
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def risk_coverage(conf: np.ndarray, correct: np.ndarray):
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    c = np.asarray(correct, float)[order]
    n = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / n
    return n / len(c), risk


def aurc(conf, correct) -> float:
    _, risk = risk_coverage(conf, correct)
    return float(risk.mean())


def selective_accuracy(conf, correct, coverage: float = 0.8) -> float:
    cov, risk = risk_coverage(conf, correct)
    k = max(1, int(np.ceil(coverage * len(cov))))
    return float(1 - risk[k - 1])


def reliability_bins(conf, correct, n_bins: int = 10):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append((float(edges[b]), float(edges[b + 1]), int(m.sum()),
                     float(conf[m].mean()) if m.any() else None,
                     float(correct[m].mean()) if m.any() else None))
    return rows


def bootstrap_ci(fn: Callable[[np.ndarray], float], n: int, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def compute_all(records: list[RunRecord], n_boot: int = 1000) -> dict:
    gold_b = np.array([r.post.gold_is_fake for r in records], bool)
    pred_b = np.array([r.final.is_fake for r in records], bool)
    conf = np.array([r.final.confidence for r in records], float)
    correct = (gold_b == pred_b).astype(float)
    gold_m = [r.post.gold_source.value for r in records]
    pred_m = [r.final.source_label.value for r in records]
    yb_t = ["fake" if g else "real" for g in gold_b]
    yb_p = ["fake" if p else "real" for p in pred_b]

    out = {
        "n": len(records),
        "parse_fail_rate": float(np.mean([not r.final.parse_ok for r in records])),
        "binary": {
            "accuracy": float(correct.mean()),
            "macro_f1": macro_f1(yb_t, yb_p, ["real", "fake"]),
            "per_class_recall": per_class_recall(yb_t, yb_p, ["real", "fake"]),
            "fake_precision": float(((pred_b & gold_b).sum()) / max(1, pred_b.sum())),
        },
        "four_way": {
            "accuracy": float(np.mean([g == p for g, p in zip(gold_m, pred_m)])),
            "macro_f1": macro_f1(gold_m, pred_m, SOURCE_LABELS),
            "per_class_recall": per_class_recall(gold_m, pred_m, SOURCE_LABELS),
        },
        "calibration": {
            "ece_15bins": ece(conf, correct, 15),
            "brier": brier(conf, correct),
            "aurc": aurc(conf, correct),
            "selective_acc_at_80_coverage": selective_accuracy(conf, correct, 0.8),
            "mean_confidence": float(conf.mean()),
        },
        "cost": {
            "prompt_tokens_per_post": float(np.mean([r.cost.prompt_tokens for r in records])),
            "completion_tokens_per_post": float(np.mean([r.cost.completion_tokens for r in records])),
            "latency_s_per_post": float(np.mean([r.cost.latency_s for r in records])),
            "llm_calls_per_post": float(np.mean([r.cost.llm_calls for r in records])),
            "search_calls_per_post": float(np.mean([r.cost.search_calls for r in records])),
        },
    }
    if n_boot and len(records) >= 20:
        n = len(records)
        out["binary"]["macro_f1_ci95"] = bootstrap_ci(
            lambda i: macro_f1([yb_t[k] for k in i], [yb_p[k] for k in i], ["real", "fake"]), n, n_boot)
        out["calibration"]["ece_ci95"] = bootstrap_ci(lambda i: ece(conf[i], correct[i], 15), n, n_boot)
    return out
