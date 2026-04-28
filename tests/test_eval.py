import numpy as np

from eval.metrics import ece


def test_ece_range():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.6, 0.3, 0.7, 0.4])
    score = ece(y_true, y_prob, n_bins=5)
    assert 0 <= score <= 1
