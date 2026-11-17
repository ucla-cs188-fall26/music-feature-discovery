from time import perf_counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from shapesearch.matching.anchors import guide_knots, match_anchors
from shapesearch.matching.dtw import BAND, dtw_align, dtw_costs
from shapesearch.matching.prep import min_range_for, normalize_feature, normalize_sketch
from shapesearch.matching.query import (SKETCH_RATE, AnchorMatch, QueryError, SearchResponse,
                                        SegmentResult, query_curves)
from shapesearch.matching.score import TAUS, combine, score_from_cost
from shapesearch.matching.windows import generate_windows

METHODS = ("euclidean", "dtw")
DEFAULT_METHOD = "euclidean"   # the method used when none is given; Part G will tell you whether to change it
MAX_PER_TRACK = 2        # at most this many results from any one track
MAX_OVERLAP = 0.5        # a result may overlap an accepted one from the same track by at most half


def _resample(rows, m):
    """Linearly resample every row (a window) to m values."""
    n = rows.shape[1]
    if n == m:
        return rows
    pos = np.linspace(0, n - 1, m)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = pos - i0
    return rows[:, i0] * (1 - frac) + rows[:, i1] * frac


def _euclidean_costs(curve, starts, n, m, feature, sketch):
    """Mean squared difference between the sketch and each window of length n
    starting at `starts`, after normalizing and resampling windows to m values."""
    stack = sliding_window_view(curve, n)[starts]          # one row per window
    stack = _resample(normalize_feature(stack, feature), m)
    return ((stack - sketch) ** 2).mean(axis=1)


def overlap_fraction(a_start, a_end, b_start, b_end):
    """How much two stretches overlap, as a fraction of the shorter one."""
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return overlap / min(a_end - a_start, b_end - b_start)


def select_top(order, track, start_s, end_s, top_k, max_per_track=MAX_PER_TRACK, max_overlap=MAX_OVERLAP):
    """Walk candidates best-first and keep one only if it adds something new:
    no more than max_per_track from one track, and no more than max_overlap
    overlap with an already-kept result from the same track.
    Returns the indices of the kept candidates."""
    kept, per_track = [], {}
    for i in order:
        t = int(track[i])
        mine = per_track.setdefault(t, [])
        if len(mine) >= max_per_track:
            continue
        if any(overlap_fraction(start_s[i], end_s[i], start_s[j], end_s[j]) > max_overlap for j in mine):
            continue
        mine.append(i)
        kept.append(i)
        if len(kept) == top_k:
            break
    return kept


def search(cache, query, method=None, band=BAND, tau=None):
    """Find the best-matching stretches of music for a sketch.

    cache   a CurveCache
    query   a Query (its mode decides between flexible and anchored)
    method  "euclidean" or "dtw" (default: DEFAULT_METHOD)
    band    DTW only: how far the path may stray from the diagonal, as a fraction of the longer curve
    tau     how quickly scores fall with cost (default: the method's value in score.TAUS)
    Returns a SearchResponse.
    """
    started = perf_counter()
    method = method or DEFAULT_METHOD
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    tau = TAUS[method] if tau is None else tau

    features = query.active_features
    unknown = [f for f in features if f not in cache.feature_names]
    if unknown:
        raise QueryError(f"unknown feature(s) {unknown}; available features: {cache.feature_names}")

    anchored = query.mode == "anchored"
    sketches = {f: normalize_sketch(c) for f, c in query_curves(query).items()}
    sketch_matrix = np.stack([sketches[f] for f in features])
    m = sketch_matrix.shape[1]
    raw = np.array([query.features[f].weight for f in features], dtype=float)
    weights = raw / raw.sum()
    min_ranges = np.array([min_range_for(f) for f in features])

    def stacked(tid):
        return np.stack([cache.curve(tid, f) for f in features])

    parts = []   # one chunk per (track, window length)
    for tid in cache.track_ids:
        info = cache.track_info(tid)
        n_cells = len(cache.curve(tid, features[0]))
        by_length = {}
        for w in generate_windows(n_cells, info["start_s"], info["end_s"], m):
            by_length.setdefault(w.n, []).append(w.start_idx)
        curves = stacked(tid) if method == "dtw" else None

        for n, starts in by_length.items():
            starts = np.array(starts)
            start_s = starts / SKETCH_RATE
            hit_t = hit_s = np.zeros((len(starts), 0))
            if anchored:
                ok, hit_t, hit_s = match_anchors(cache.hits(tid), query.anchors_s, query.duration_s,
                                                 start_s, np.full(len(starts), n / SKETCH_RATE))
                if not ok.any():
                    continue
                starts, start_s, hit_t, hit_s = starts[ok], start_s[ok], hit_t[ok], hit_s[ok]

            if method == "euclidean":
                costs = np.stack([_euclidean_costs(cache.curve(tid, f), starts, n, m, f, sketches[f])
                                  for f in features])                       # one row per feature
            else:
                knots_i = knots_j = None
                if anchored:
                    knots_i, knots_j = guide_knots(query.anchors_s, hit_t, start_s, m, n)
                costs, _ = dtw_costs(curves, starts, n, sketch_matrix, weights, min_ranges, band, knots_i, knots_j)
                costs = costs.T
            scores = score_from_cost(costs, tau)
            parts.append({"track": np.full(len(starts), tid), "start_idx": starts, "n": np.full(len(starts), n),
                          "start_s": start_s, "end_s": start_s + n / SKETCH_RATE,
                          "overall": combine(scores, weights), "scores": scores,
                          "hit_t": hit_t, "hit_s": hit_s})

    results = []
    if parts:
        joined = {k: np.concatenate([p[k] for p in parts], axis=1 if k == "scores" else 0) for k in parts[0]}
        order = np.argsort(-joined["overall"], kind="stable")
        for i in select_top(order, joined["track"], joined["start_s"], joined["end_s"], query.top_k):
            tid, n = int(joined["track"][i]), int(joined["n"][i])
            anchors = [AnchorMatch(float(a), float(joined["hit_t"][i, k]), float(joined["hit_s"][i, k]))
                       for k, a in enumerate(query.anchors_s)]
            path = []
            if method == "dtw":
                knots_i = knots_j = None
                if anchored:
                    knots_i, knots_j = guide_knots(query.anchors_s, joined["hit_t"][i:i + 1],
                                                   joined["start_s"][i:i + 1], m, n)
                    knots_j = knots_j[0]
                pairs, _ = dtw_align(stacked(tid), int(joined["start_idx"][i]), n, sketch_matrix, weights,
                                     min_ranges, band, knots_i, knots_j)
                path = [(int(a), int(b)) for a, b in pairs]
            results.append(SegmentResult(
                track_id=tid, start_s=float(joined["start_s"][i]), end_s=float(joined["end_s"][i]),
                overall_score=float(joined["overall"][i]),
                feature_scores={f: float(joined["scores"][k, i]) for k, f in enumerate(features)},
                anchors=anchors, path=path))

    used = {f: float(w) for f, w in zip(features, weights)}
    used.update({f: 0.0 for f in query.features if f not in used})   # "don't care" features, shown as 0
    return SearchResponse(results, used, query.mode, perf_counter() - started)