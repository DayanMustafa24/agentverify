import numpy as np
from agentverify.eval.metrics import ece, brier, macro_f1, aurc, selective_accuracy


def test_perfectly_calibrated_has_zero_ece():
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2)
    assert ece(conf, correct) < 1e-9


def test_overconfident_ece():
    conf = np.array([0.95] * 10)
    correct = np.array([1] * 5 + [0] * 5)
    assert abs(ece(conf, correct) - 0.45) < 1e-9


def test_brier_bounds():
    assert brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert brier(np.array([1.0, 0.0]), np.array([0, 1])) == 1.0


def test_macro_f1_perfect_and_worst():
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "b"], ["b", "a"], ["a", "b"]) == 0.0


def test_selective_accuracy_improves_with_good_confidence():
    conf = np.array([0.9, 0.8, 0.7, 0.2])
    correct = np.array([1, 1, 1, 0])
    assert selective_accuracy(conf, correct, 0.75) == 1.0
    assert aurc(conf, correct) < aurc(conf[::-1], correct)
