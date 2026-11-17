import numpy as np

# How quickly scores fall as the difference grows. S = 100 * exp(-cost / tau).
# On normalized (0-1) curves, with tau = 0.1 a cost of 0.01 scores about 90,
# 0.02 about 82 and 0.1 about 37. The two methods get their own value because
# DTW's costs are lower than the baseline's for the same match; calibrate both
# in Part G so strong matches land around 80 to 95.
TAUS = {"euclidean": 0.1, "dtw": 0.03}
TAU = TAUS["euclidean"]     # the baseline's value; kept so older code keeps working


def score_from_cost(cost, tau=TAU):
    """cost: the mean squared difference between a sketch and a window, for one
    feature. Returns a score from 0 (nothing alike) to 100 (identical)."""
    return 100.0 * np.exp(-np.asarray(cost, dtype=np.float64) / tau)


def combine(feature_scores, weights):
    """Weighted average of per-feature scores.
    feature_scores has one row per feature; weights has one entry per feature."""
    w = np.asarray(weights, dtype=np.float64)
    return np.tensordot(w, feature_scores, axes=(0, 0)) / w.sum()