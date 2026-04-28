from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auroc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece_val += (m.sum() / len(y_true)) * abs(acc - conf)
    return float(ece_val)
