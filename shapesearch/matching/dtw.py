import math

import numpy as np

try:
    from numba import njit, prange
except ImportError:   # everything still works without numba, but far too slowly to search a catalog
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function
else:
    HAVE_NUMBA = True

BAND = 0.15   # how far the path may stray from the diagonal, as a fraction of the longer curve


# ------------------------------------------------------------ small helpers

@njit(cache=True)
def _normalize_window(curves, start, n, min_ranges, out):
    """Copy curves[:, start:start+n] into out, rescaled to 0-1 per feature with the
    flatness guard. Same formula as prep.normalize()."""
    for f in range(curves.shape[0]):
        lo = np.inf
        hi = -np.inf
        for j in range(n):
            v = float(curves[f, start + j])
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        scale = max(hi - lo, min_ranges[f])
        for j in range(n):
            out[f, j] = (float(curves[f, start + j]) - lo) / scale


@njit(cache=True)
def _fill_centre(knots_i, knots_j, centre):
    """The line the band follows: straight lines through the knots, where knot k
    pairs sketch value knots_i[k] with window value knots_j[k]."""
    for k in range(len(knots_i) - 1):
        i0 = knots_i[k]
        i1 = knots_i[k + 1]
        j0 = knots_j[k]
        j1 = knots_j[k + 1]
        for i in range(i0, i1 + 1):
            centre[i] = j0 + (j1 - j0) * (i - i0) / (i1 - i0)


@njit(cache=True)
def _half_width(centre, m, n, band_frac):
    """Half the band's width, in cells. Never so narrow that a steep stretch of the
    centre line could leave a row without any reachable cell."""
    width = band_frac * max(m, n)
    steepest = 0.0
    for i in range(1, m):
        steepest = max(steepest, abs(centre[i] - centre[i - 1]))
    return max(width, (steepest + 1.0) / 2.0 + 1.0)


# ---------------------------------------------- many windows, costs only

@njit(cache=True, parallel=True)
def _batch(curves, starts, n, sketch, weights, min_ranges, band_frac, knots_i, knots_j, costs, lengths):
    F, m = sketch.shape
    for w in prange(starts.shape[0]):
        x = np.empty((F, n))
        _normalize_window(curves, starts[w], n, min_ranges, x)
        centre = np.empty(m)
        _fill_centre(knots_i, knots_j[w], centre)
        half = _half_width(centre, m, n, band_frac)

        # Two rows of the grid at a time. Alongside D we carry, for the best path
        # into each cell, its length and each feature's summed squared difference.
        Dp = np.full(n, np.inf)
        Dc = np.full(n, np.inf)
        Ap = np.zeros((F, n))
        Ac = np.zeros((F, n))
        Lp = np.zeros(n)
        Lc = np.zeros(n)
        lo_p = 0
        hi_p = -1
        for i in range(m):
            lo = max(0, int(math.ceil(centre[i] - half)))
            hi = min(n - 1, int(math.floor(centre[i] + half)))
            for j in range(lo, hi + 1):
                if i == 0 and j == 0:
                    base = 0.0
                    src = 3
                else:
                    base = np.inf
                    src = -1
                    if i > 0:
                        if j > 0 and lo_p <= j - 1 and j - 1 <= hi_p and Dp[j - 1] < base:
                            base = Dp[j - 1]
                            src = 0                 # diagonal wins ties
                        if lo_p <= j and j <= hi_p and Dp[j] < base:
                            base = Dp[j]
                            src = 1                 # from the row above
                    if j > lo and Dc[j - 1] < base:
                        base = Dc[j - 1]
                        src = 2                     # from the cell to the left
                    if src < 0:
                        Dc[j] = np.inf
                        continue

                total = 0.0
                for f in range(F):
                    d = sketch[f, i] - x[f, j]
                    c = d * d
                    total += weights[f] * c
                    if src == 0:
                        Ac[f, j] = Ap[f, j - 1] + c
                    elif src == 1:
                        Ac[f, j] = Ap[f, j] + c
                    elif src == 2:
                        Ac[f, j] = Ac[f, j - 1] + c
                    else:
                        Ac[f, j] = c
                if src == 0:
                    Lc[j] = Lp[j - 1] + 1.0
                elif src == 1:
                    Lc[j] = Lp[j] + 1.0
                elif src == 2:
                    Lc[j] = Lc[j - 1] + 1.0
                else:
                    Lc[j] = 1.0
                Dc[j] = base + total

            lo_p = lo
            hi_p = hi
            Dp, Dc = Dc, Dp
            Ap, Ac = Ac, Ap
            Lp, Lc = Lc, Lp

        if Dp[n - 1] < np.inf:
            for f in range(F):
                costs[w, f] = Ap[f, n - 1] / Lp[n - 1]
            lengths[w] = Lp[n - 1]
        else:
            for f in range(F):
                costs[w, f] = np.inf
            lengths[w] = 0.0


# --------------------------------------------- one window, with the path

@njit(cache=True)
def _align(curves, start, n, sketch, weights, min_ranges, band_frac, knots_i, knots_j):
    F, m = sketch.shape
    x = np.empty((F, n))
    _normalize_window(curves, start, n, min_ranges, x)
    centre = np.empty(m)
    _fill_centre(knots_i, knots_j, centre)
    half = _half_width(centre, m, n, band_frac)

    D = np.full((m, n), np.inf)          # the whole grid, so the path can be traced back
    for i in range(m):
        lo = max(0, int(math.ceil(centre[i] - half)))
        hi = min(n - 1, int(math.floor(centre[i] + half)))
        for j in range(lo, hi + 1):
            if i == 0 and j == 0:
                base = 0.0
            else:
                base = np.inf
                if i > 0 and j > 0 and D[i - 1, j - 1] < base:
                    base = D[i - 1, j - 1]
                if i > 0 and D[i - 1, j] < base:
                    base = D[i - 1, j]
                if j > 0 and D[i, j - 1] < base:
                    base = D[i, j - 1]
                if base == np.inf:
                    continue
            total = 0.0
            for f in range(F):
                d = sketch[f, i] - x[f, j]
                total += weights[f] * d * d
            D[i, j] = base + total

    path = np.empty((m + n, 2), dtype=np.int64)
    i = m - 1
    j = n - 1
    path[0, 0] = i
    path[0, 1] = j
    count = 1
    while i > 0 or j > 0:
        best = np.inf
        step = -1
        if i > 0 and j > 0 and D[i - 1, j - 1] < best:
            best = D[i - 1, j - 1]
            step = 0
        if i > 0 and D[i - 1, j] < best:
            best = D[i - 1, j]
            step = 1
        if j > 0 and D[i, j - 1] < best:
            best = D[i, j - 1]
            step = 2
        if step < 0:
            raise ValueError("no path through the band")
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
        path[count, 0] = i
        path[count, 1] = j
        count += 1

    costs = np.zeros(F)
    for k in range(count):
        for f in range(F):
            d = sketch[f, path[k, 0]] - x[f, path[k, 1]]
            costs[f] += d * d
    for f in range(F):
        costs[f] /= count
    return path[:count][::-1].copy(), costs


# ----------------------------------------------------------- public side

def _arrays(curves, sketch, weights, min_ranges):
    curves = np.ascontiguousarray(curves, dtype=np.float32)
    sketch = np.ascontiguousarray(sketch, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    min_ranges = np.ascontiguousarray(min_ranges, dtype=np.float64)
    if curves.ndim != 2 or sketch.ndim != 2 or curves.shape[0] != sketch.shape[0]:
        raise ValueError("curves and sketch must both have one row per feature")
    if len(weights) != sketch.shape[0] or len(min_ranges) != sketch.shape[0]:
        raise ValueError("weights and min_ranges need one value per feature")
    return curves, sketch, weights, min_ranges


def _default_knots(m, n):
    return np.array([0, m - 1], dtype=np.int64), np.array([0.0, n - 1.0])


def dtw_costs(curves, starts, n, sketch, weights, min_ranges, band=BAND, knots_i=None, knots_j=None):
    """Align the sketch with many windows of the same length at once.

    curves      (features, cells): one track's curves
    starts      first cell of each window
    n           number of cells in every window
    sketch      (features, m), already normalized
    weights     one per feature (they shape the path; use ones summing to 1)
    min_ranges  flatness guard per feature, for normalizing each window
    knots_i     optional (K,): sketch cells that must line up with window cells ...
    knots_j     ... optional (windows, K): the window cells they line up with
                (anchored mode; omitted means the plain straight diagonal)

    Returns (costs, lengths): costs is (windows, features), each feature's average
    squared difference along the shared path; lengths is the path length.
    """
    curves, sketch, weights, min_ranges = _arrays(curves, sketch, weights, min_ranges)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    m = sketch.shape[1]
    if n < 2 or m < 2:
        raise ValueError("windows and sketches need at least 2 values")
    if len(starts) and (starts.min() < 0 or starts.max() + n > curves.shape[1]):
        raise ValueError("a window runs off the end of the curves")
    if knots_i is None:
        knots_i, row = _default_knots(m, n)
        knots_j = np.tile(row, (len(starts), 1))
    knots_i = np.ascontiguousarray(knots_i, dtype=np.int64)
    knots_j = np.ascontiguousarray(knots_j, dtype=np.float64)

    costs = np.empty((len(starts), sketch.shape[0]))
    lengths = np.empty(len(starts))
    if len(starts):
        _batch(curves, starts, n, sketch, weights, min_ranges, float(band), knots_i, knots_j, costs, lengths)
    return costs, lengths


def dtw_align(curves, start, n, sketch, weights, min_ranges, band=BAND, knots_i=None, knots_j=None):
    """Align the sketch with ONE window and return the path.

    Same inputs as dtw_costs, for a single window starting at cell `start`
    (knots_j is then a single row of K values).

    Returns (path, costs): path is an array of (sketch cell, window cell) pairs,
    from (0, 0) to (m-1, n-1), where window cell 0 is the window's first cell;
    costs holds each feature's average squared difference along the path.
    """
    curves, sketch, weights, min_ranges = _arrays(curves, sketch, weights, min_ranges)
    m = sketch.shape[1]
    if start < 0 or start + n > curves.shape[1]:
        raise ValueError("the window runs off the end of the curves")
    if knots_i is None:
        knots_i, knots_j = _default_knots(m, n)
    knots_i = np.ascontiguousarray(knots_i, dtype=np.int64)
    knots_j = np.ascontiguousarray(knots_j, dtype=np.float64)
    path, costs = _align(curves, int(start), int(n), sketch, weights, min_ranges, float(band), knots_i, knots_j)
    return path, costs