import numpy as np

from shapesearch.matching.query import SKETCH_RATE

ANCHOR_TOL_S = 0.5   # how far from the expected position a hit may be (half the 1 s step between windows)


def match_anchors(hits, anchors_s, duration_s, start_s, length_s, tol_s=ANCHOR_TOL_S):
    """Find, for every window, the hit point that satisfies each anchor.

    hits         [(time_s, strength), ...] stored for one track
    anchors_s    anchor times in the sketch, increasing
    duration_s   the sketch's length in seconds
    start_s      (windows,) start time of each window in the track
    length_s     (windows,) length of each window in seconds

    Returns (ok, hit_times, strengths):
      ok          (windows,) True if every anchor found its own hit, with the hits in
                  the same order as the anchors (two anchors may not share a hit)
      hit_times   (windows, anchors) the chosen hit's time, in the track
      strengths   (windows, anchors) its strength; only meaningful where ok is True
    """
    start_s = np.asarray(start_s, dtype=np.float64)
    length_s = np.asarray(length_s, dtype=np.float64)
    anchors = np.asarray(anchors_s, dtype=np.float64)
    count, wanted = len(start_s), len(anchors)
    if not len(hits):
        return np.zeros(count, dtype=bool), np.zeros((count, wanted)), np.zeros((count, wanted))

    order = np.argsort([h[0] for h in hits], kind="stable")
    times = np.array([hits[i][0] for i in order], dtype=np.float64)
    strengths = np.array([hits[i][1] for i in order], dtype=np.float64)

    expected = start_s[:, None] + anchors[None, :] * (length_s / duration_s)[:, None]   # (windows, anchors)
    gap = np.abs(expected[:, :, None] - times[None, None, :])                           # (windows, anchors, hits)
    nearest = gap.argmin(axis=2)
    close = np.take_along_axis(gap, nearest[:, :, None], axis=2)[:, :, 0] <= tol_s + 1e-9

    ok = close.all(axis=1)
    if wanted > 1:
        ok &= (np.diff(nearest, axis=1) > 0).all(axis=1)       # each anchor its own hit, in order
    return ok, times[nearest], strengths[nearest]


def guide_knots(anchors_s, hit_times, start_s, m, n, rate=SKETCH_RATE):
    """Turn the anchor pairings into the lines DTW's band should follow.

    Each anchor pairs a sketch cell with the window cell holding its hit. The band
    then follows straight lines (0,0) -> pairing -> ... -> (m-1, n-1) instead of the
    plain diagonal, so the path is steered through the anchors.

    hit_times  (windows, anchors) as returned by match_anchors, for the windows kept
    start_s    (windows,) their start times
    m, n       sketch length and window length, in cells

    Returns (knots_i, knots_j): knots_i is (K,) and shared; knots_j is (windows, K).
    """
    start_s = np.asarray(start_s, dtype=np.float64)
    i_cells, columns, seen = [], [], set()
    for k, a in enumerate(anchors_s):
        i = int(np.clip(np.floor(a * rate), 0, m - 1))
        if i in (0, m - 1) or i in seen:      # the ends are already fixed; one knot per sketch cell
            continue
        seen.add(i)
        i_cells.append(i)
        columns.append(k)
    knots_i = np.array([0] + i_cells + [m - 1], dtype=np.int64)
    inner = np.floor((hit_times[:, columns] - start_s[:, None]) * rate)
    inner = np.clip(inner, 0, n - 1)
    zero = np.zeros((len(start_s), 1))
    last = np.full((len(start_s), 1), n - 1.0)
    knots_j = np.hstack([zero, inner, last]) if columns else np.hstack([zero, last])
    return knots_i, knots_j