"""Platt scaling of the verdict confidence. Fit on the calibration set only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


class Platt:
    def __init__(self):
        self.lr, self.identity = None, True

    def fit(self, conf, correct):
        y = np.asarray(correct, int)
        if len(y) < 10 or len(set(y)) < 2:
            self.identity = True
            return self
        self.lr = LogisticRegression(C=1e6).fit(_logit(conf)[:, None], y)
        self.identity = False
        return self

    def apply(self, conf):
        c = np.asarray(conf, float)
        if self.identity:
            return c
        return self.lr.predict_proba(_logit(c)[:, None])[:, 1]
