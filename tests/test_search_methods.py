import json

import numpy as np
import pytest

from shapesearch.matching.query import Query
from shapesearch.matching.search import search
from synthetic import (N, background, build_shape, fade_shape, make_cache, plant, single_peak_shape,
                       two_peak_shape)

BUILD_QUERY = {"features": {"loudness": {"points": [[0, 0.1], [20, 0.9], [30, 0.9]]},
                            "brightness": {"points": [[0, 0.1], [30, 0.9]]}}, "duration_s": 30}


def anchored_query(**changes):
    body = dict(BUILD_QUERY, mode="anchored", anchors_s=[8])
    body.update(changes)
    return Query.from_dict(body)


# ---- DTW ----

def test_dtw_finds_the_planted_segment_first_and_returns_its_path():
    rng = np.random.default_rng(0)
    cache = make_cache({0: (*background(rng), 0.0, 120.0), 1: (*background(rng), 0.0, 120.0),
                        2: plant(rng, build_shape(), at_s=40), 3: plant(rng, fade_shape(), at_s=40)})
    response = search(cache, Query.from_dict(BUILD_QUERY), method="dtw")
    best, runner_up = response.results[0], response.results[1]
    assert best.track_id == 2 and abs(best.start_s - 40) <= 1 and best.overall_score > 90
    assert best.overall_score - runner_up.overall_score > 20
    n = round((best.end_s - best.start_s) * 2)
    assert best.path[0] == (0, 0) and best.path[-1] == (59, n - 1)           # corner to corner, in half-second cells
    assert all(0 <= i < 60 and 0 <= j < n for i, j in best.path)


TWO_PEAKS = [[0, 0], [6, 0], [8, 1], [10, 0], [20, 0], [22, 1], [24, 0], [30, 0]]    # peaks at 8 s and 22 s
TWO_PEAKS_QUERY = {"features": {"loudness": {"points": TWO_PEAKS}, "brightness": {"points": TWO_PEAKS}}, "duration_s": 30}


def test_dtw_copes_with_timing_that_is_off_where_the_baseline_does_not():
    rng = np.random.default_rng(1)
    # the music's first peak is at 12 s, not 8 s: no shift or stretch of the whole window can line up both peaks
    cache = make_cache({0: (*background(rng), 0.0, 120.0), 1: plant(rng, two_peak_shape(12, 22), at_s=50),
                        2: plant(rng, single_peak_shape(15), at_s=50)})
    query = Query.from_dict(TWO_PEAKS_QUERY)
    dtw = search(cache, query, method="dtw", tau=0.1).results[0]        # same tau for both, so the scores compare
    rigid = search(cache, query, method="euclidean", tau=0.1).results[0]
    assert dtw.track_id == 1 and abs(dtw.start_s - 50) <= 1 and dtw.overall_score > 90
    assert rigid.overall_score < dtw.overall_score - 20


def test_dtw_still_tells_a_different_shape_apart():
    rng = np.random.default_rng(1)
    cache = make_cache({1: plant(rng, two_peak_shape(12, 22), at_s=50), 2: plant(rng, single_peak_shape(15), at_s=50)})
    results = search(cache, Query.from_dict(TWO_PEAKS_QUERY), method="dtw").results
    best_per_track = {}
    for r in results:
        best_per_track.setdefault(r.track_id, r.overall_score)
    assert best_per_track[1] > best_per_track[2] + 30                     # the two-peak music clearly beats the one-peak music


def test_dtw_finds_a_time_stretched_copy():
    rng = np.random.default_rng(2)
    cache = make_cache({0: (*background(rng), 0.0, 120.0), 1: plant(rng, build_shape(seconds=24), at_s=50)})
    best = search(cache, Query.from_dict(BUILD_QUERY), method="dtw").results[0]
    assert best.track_id == 1 and abs(best.start_s - 50) <= 1 and best.end_s - best.start_s == pytest.approx(24, abs=0.5)


def test_dtw_zero_weight_ignores_a_feature():
    rng = np.random.default_rng(3)
    good = plant(rng, build_shape(), 40)
    wrong_brightness = (good[0], np.full(N, 9.0) + rng.normal(0, 0.02, N), 0.0, 120.0)
    cache = make_cache({0: good, 1: wrong_brightness})
    both = search(cache, Query.from_dict(BUILD_QUERY), method="dtw")
    assert [r.track_id for r in both.results[:2]] == [0, 1] and both.results[0].overall_score > both.results[1].overall_score + 10
    loud_only = Query.from_dict(dict(BUILD_QUERY, features={**BUILD_QUERY["features"],
                                                            "brightness": {"points": [[0, 0.1], [30, 0.9]], "weight": 0}}))
    response = search(cache, loud_only, method="dtw")
    assert set(response.results[0].feature_scores) == {"loudness"}
    assert response.weights == {"loudness": 1.0, "brightness": 0.0}
    assert abs(response.results[0].overall_score - response.results[1].overall_score) < 5


def test_dtw_never_returns_the_silent_intro_and_the_response_is_json_ready():
    rng = np.random.default_rng(4)
    loud, bright, _, _ = plant(rng, build_shape(), at_s=10)
    cache = make_cache({0: (loud, bright, 70.0, 120.0)})
    response = search(cache, Query.from_dict(BUILD_QUERY), method="dtw")
    assert response.results and all(r.start_s >= 70 for r in response.results)
    json.dumps(response.to_dict())


def test_the_band_setting_limits_how_far_the_path_may_stray():
    rng = np.random.default_rng(1)
    cache = make_cache({1: plant(rng, two_peak_shape(12, 22), at_s=50)})           # the first peak is 4 s (8 cells) late
    query = Query.from_dict(TWO_PEAKS_QUERY)

    def stray(result):
        n = round((result.end_s - result.start_s) * 2)
        return max(abs(j - i * (n - 1) / 59) for i, j in result.path)

    narrow = search(cache, query, method="dtw", band=0.05).results[0]
    wide = search(cache, query, method="dtw", band=0.4).results[0]
    assert stray(narrow) <= 4
    assert wide.overall_score >= narrow.overall_score - 1e-9


def test_unknown_method_is_a_clear_error():
    cache = make_cache({0: plant(np.random.default_rng(0), build_shape(), 40)})
    with pytest.raises(ValueError, match=r"method must be one of \['euclidean', 'dtw'\]"):
        search(cache, Query.from_dict(BUILD_QUERY), method="fft")


# ---- anchored mode ----

def anchored_catalog():
    rng = np.random.default_rng(5)
    tracks = {0: plant(rng, build_shape(), 40),                 # the shape, with a hit exactly where the anchor says
              1: plant(rng, build_shape(), 40),                 # the same shape but no hits stored
              2: plant(rng, build_shape(), 40),                 # the same shape but its hit is somewhere else
              3: (*background(rng), 0.0, 120.0)}                # a hit in the right place, but no matching shape
    hits = {0: [(48.0, 7.0)], 2: [(75.0, 6.0)], 3: [(48.0, 5.0)]}
    return make_cache(tracks, hits)


@pytest.mark.parametrize("method", ["euclidean", "dtw"])
def test_anchored_mode_only_returns_windows_with_a_hit_in_the_right_place(method):
    cache = anchored_catalog()
    flexible = search(cache, Query.from_dict(BUILD_QUERY), method=method)
    anchored = search(cache, anchored_query(), method=method)
    assert {0, 1, 2} <= {r.track_id for r in flexible.results}              # without anchors, tracks 1 and 2 are found too
    assert anchored.mode == "anchored"
    assert 1 not in {r.track_id for r in anchored.results}                   # no hits stored, so never a candidate
    assert all(r.start_s >= 66 for r in anchored.results if r.track_id == 2)  # its only hit is at 75 s, so only windows near there
    best = anchored.results[0]
    assert best.track_id == 0 and best.start_s == 40 and best.overall_score > 80
    (match,) = best.anchors
    assert (match.anchor_s, match.hit_time_s, match.hit_strength) == (8.0, 48.0, 7.0)
    assert all(r.start_s == pytest.approx(r.anchors[0].hit_time_s - 8 * (r.end_s - r.start_s) / 30, abs=0.51)
               for r in anchored.results)


def test_anchored_dtw_passes_the_path_through_the_anchor():
    best = search(anchored_catalog(), anchored_query(), method="dtw").results[0]
    anchor_cell = 16                                                         # 8 s * 2 cells per second
    assert any(i == anchor_cell and abs(j - 16) <= 3 for i, j in best.path)


def test_anchored_mode_scales_the_expected_hit_with_the_window_length():
    rng = np.random.default_rng(6)
    cache = make_cache({0: plant(rng, build_shape(seconds=24), at_s=50)}, {0: [(56.4, 6.0)]})   # 50 s + 8 s * 24/30
    best = search(cache, anchored_query()).results[0]
    assert best.start_s == 50 and best.end_s - best.start_s == pytest.approx(24, abs=0.5)
    assert best.anchors[0].hit_time_s == 56.4


def test_anchored_search_with_no_hits_anywhere_returns_nothing():
    rng = np.random.default_rng(7)
    cache = make_cache({0: plant(rng, build_shape(), 40), 1: plant(rng, build_shape(), 40)})
    for method in ("euclidean", "dtw"):
        response = search(cache, anchored_query(), method=method)
        assert response.results == [] and response.mode == "anchored"


def test_two_anchors_need_two_different_hits_in_order():
    rng = np.random.default_rng(8)
    cache = make_cache({0: plant(rng, build_shape(), 40), 1: plant(rng, build_shape(), 40)},
                       {0: [(48.0, 7.0), (58.0, 5.0)], 1: [(48.0, 7.0)]})
    two = Query.from_dict(dict(BUILD_QUERY, mode="anchored", anchors_s=[8, 18]))
    for method in ("euclidean", "dtw"):
        results = search(cache, two, method=method).results
        assert {r.track_id for r in results} == {0}
        assert [(a.anchor_s, a.hit_time_s) for a in results[0].anchors] == [(8.0, 48.0), (18.0, 58.0)]