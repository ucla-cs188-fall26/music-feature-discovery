import numpy as np
import pytest

from shapesearch.matching.dtw import BAND, dtw_align, dtw_costs
from shapesearch.matching.prep import normalize

MIN_RANGES = np.array([3.0, 0.3])


def reference(sketch, window, weights, band):
    """Plain, slow DTW on the full grid, written straight from the recurrence. Returns D."""
    m, n = sketch.shape[1], window.shape[1]
    centre = np.arange(m) * (n - 1) / (m - 1)
    half = max(band * max(m, n), (np.abs(np.diff(centre)).max() + 1) / 2 + 1)
    D = np.full((m, n), np.inf)
    for i in range(m):
        for j in range(n):
            if abs(j - centre[i]) > half + 1e-9:
                continue
            cost = sum(weights[f] * (sketch[f, i] - window[f, j]) ** 2 for f in range(len(weights)))
            if i == 0 and j == 0:
                D[i, j] = cost
                continue
            before = min(D[i - 1, j - 1] if i and j else np.inf, D[i - 1, j] if i else np.inf,
                         D[i, j - 1] if j else np.inf)
            D[i, j] = before + cost
    return D


def normalized(curves, start, n):
    return np.stack([normalize(curves[f, start:start + n].astype(np.float64), MIN_RANGES[f])
                     for f in range(curves.shape[0])])


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    curves = rng.normal(size=(2, 300)).cumsum(axis=1).astype(np.float32)
    return curves, rng.random((2, 40)), np.array([0.5, 0.5])


def test_matches_a_slow_reference_and_the_path_adds_up_to_it(data):
    curves, sketch, w = data
    for start, n in [(0, 40), (25, 44), (200, 32)]:
        costs, lengths = dtw_costs(curves, [start], n, sketch, w, MIN_RANGES)
        path, path_costs = dtw_align(curves, start, n, sketch, w, MIN_RANGES)
        x = normalized(curves, start, n)
        best = reference(sketch, x, w, BAND)[-1, -1]
        along = sum(w[f] * ((sketch[f, path[:, 0]] - x[f, path[:, 1]]) ** 2).sum() for f in range(2))
        assert along == pytest.approx(best)                                # the path really is the cheapest one
        assert path_costs == pytest.approx(costs[0])                       # both routes give the same per-feature costs
        assert lengths[0] == len(path)
        assert costs[0] @ w * len(path) == pytest.approx(best)             # and they add up to the grid's answer


def test_path_starts_and_ends_in_the_corners_and_only_moves_forward(data):
    curves, sketch, w = data
    path, _ = dtw_align(curves, 10, 46, sketch, w, MIN_RANGES)
    assert path[0].tolist() == [0, 0] and path[-1].tolist() == [39, 45]
    steps = {tuple(s) for s in np.diff(path, axis=0)}
    assert steps <= {(1, 1), (1, 0), (0, 1)}


def test_path_stays_inside_the_band(data):
    curves, sketch, w = data
    n, band = 46, 0.10
    path, _ = dtw_align(curves, 10, n, sketch, w, MIN_RANGES, band=band)
    centre = path[:, 0] * (n - 1) / (sketch.shape[1] - 1)
    assert np.abs(path[:, 1] - centre).max() <= max(band * n, 3.0) + 1e-9


def test_a_window_identical_to_the_sketch_costs_nothing():
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(2, 100)).cumsum(axis=1).astype(np.float32)
    sketch = normalized(curves, 20, 40)
    costs, _ = dtw_costs(curves, [20], 40, sketch, np.array([0.5, 0.5]), MIN_RANGES)
    assert costs[0] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_bending_time_lowers_the_cost_of_a_stretched_copy():
    t = np.linspace(0, 1, 60)
    sketch = np.interp(t, [0, 0.4, 1], [0.0, 1.0, 1.0])[None]               # peak reached 40% of the way
    window = np.interp(t, [0, 0.6, 1], [0.0, 1.0, 1.0])[None].astype(np.float32) * 10   # peak reached 60% of the way
    w, mr = np.array([1.0]), np.array([1.0])
    bent, _ = dtw_costs(window, [0], 60, sketch, w, mr)
    rigid = ((normalize(window[0], 1.0) - sketch[0]) ** 2).mean()
    assert bent[0, 0] < rigid / 5


def test_a_wider_band_never_costs_more_in_total():
    rng = np.random.default_rng(5)
    curves = rng.normal(size=(1, 200)).cumsum(axis=1).astype(np.float32)
    sketch, w, mr = rng.random((1, 40)), np.array([1.0]), np.array([1.0])
    totals = []
    for band in (0.05, 0.15, 0.4, 1.0):
        path, _ = dtw_align(curves, 30, 44, sketch, w, mr, band=band)
        x = normalize(curves[0, 30:74].astype(np.float64), 1.0)
        totals.append(((sketch[0, path[:, 0]] - x[path[:, 1]]) ** 2).sum())
    assert all(a >= b - 1e-12 for a, b in zip(totals, totals[1:]))


def test_weights_steer_the_shared_path():
    rng = np.random.default_rng(7)
    curves = rng.normal(size=(2, 200)).cumsum(axis=1).astype(np.float32)
    sketch = rng.random((2, 40))

    def total(weights, feature):
        path, costs = dtw_align(curves, 20, 44, sketch, np.array(weights), MIN_RANGES)
        return costs[feature] * len(path)

    assert total([1.0, 0.0], 0) <= total([0.5, 0.5], 0) + 1e-9           # the path that favours feature 0 is best for it
    assert total([0.0, 1.0], 1) <= total([0.5, 0.5], 1) + 1e-9


def test_many_windows_at_once_match_one_at_a_time(data):
    curves, sketch, w = data
    starts = np.array([0, 7, 50, 120, 250])
    costs, lengths = dtw_costs(curves, starts, 44, sketch, w, MIN_RANGES)
    for k, s in enumerate(starts):
        path, one = dtw_align(curves, int(s), 44, sketch, w, MIN_RANGES)
        assert costs[k] == pytest.approx(one) and lengths[k] == len(path)


def test_guiding_knots_steer_the_path_through_an_anchor():
    m, n = 60, 66
    sketch = np.zeros((1, m))
    sketch[0, 28:32] = 1.0                                                  # a spike at sketch cells 28-31
    window = np.zeros((1, 200), dtype=np.float32)
    window[0, 48:52] = 1.0                                                  # the same spike, far from the diagonal
    w, mr = np.array([1.0]), np.array([0.5])
    plain, _ = dtw_costs(window, [0], n, sketch, w, mr)
    guided, _ = dtw_costs(window, [0], n, sketch, w, mr,
                          knots_i=np.array([0, 30, m - 1]), knots_j=np.array([[0.0, 49.0, n - 1.0]]))
    assert plain[0, 0] > 0.03 and guided[0, 0] < 0.005
    path, _ = dtw_align(window, 0, n, sketch, w, mr, knots_i=np.array([0, 30, m - 1]), knots_j=np.array([0.0, 49.0, n - 1.0]))
    assert any(i == 30 and 48 <= j <= 51 for i, j in path)


def test_bad_inputs_are_clear_errors(data):
    curves, sketch, w = data
    with pytest.raises(ValueError, match="runs off the end"):
        dtw_costs(curves, [280], 44, sketch, w, MIN_RANGES)
    with pytest.raises(ValueError, match="one row per feature"):
        dtw_costs(curves[:1], [0], 44, sketch, w, MIN_RANGES)
    with pytest.raises(ValueError, match="one value per feature"):
        dtw_costs(curves, [0], 44, sketch, np.array([1.0]), MIN_RANGES)