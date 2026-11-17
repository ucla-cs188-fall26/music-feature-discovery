import numpy as np
import pytest

from shapesearch.matching.prep import CurveCache
from shapesearch.matching.query import Query, QueryError
from shapesearch.matching.search import overlap_fraction, search, select_top

RNG = np.random.default_rng(0)
N = 240                                   # 120 s at 2 values per second


def build_shape(seconds=30):
    """A slow build then a sustained peak, in loudness, with brightness rising too."""
    t = np.linspace(0, seconds, int(seconds * 2))
    return np.interp(t, [0, 0.67 * seconds, seconds], [-30, -12, -12]), np.interp(t, [0, seconds], [8.5, 10.5])


def fade_shape(seconds=30):
    t = np.linspace(0, seconds, int(seconds * 2))
    return np.interp(t, [0, seconds], [-12, -30]), np.interp(t, [0, seconds], [10.5, 8.5])


def background():
    """Nearly flat noise: varies by less than the flatness guard."""
    return -20 + RNG.normal(0, 0.3, N), 9 + RNG.normal(0, 0.03, N)


def make_cache(tracks):
    """tracks: {id: (loudness, brightness, start_s, end_s)}"""
    curves, info = {}, {}
    for tid, (loud, bright, start, end) in tracks.items():
        curves[(tid, "loudness")] = np.asarray(loud, dtype=np.float32)
        curves[(tid, "brightness")] = np.asarray(bright, dtype=np.float32)
        info[tid] = {"start_s": start, "end_s": end, "duration_s": len(loud) / 2}
    return CurveCache(curves, info, {t: [] for t in tracks}, ["loudness", "brightness"], {})


def plant(shape, at_s):
    loud, bright = background()
    i = int(at_s * 2)
    loud[i:i + len(shape[0])] = shape[0]
    bright[i:i + len(shape[1])] = shape[1]
    return loud, bright, 0.0, N / 2


BUILD_QUERY = {"features": {"loudness": {"points": [[0, 0.1], [20, 0.9], [30, 0.9]]},
                            "brightness": {"points": [[0, 0.1], [30, 0.9]]}}, "duration_s": 30}


def test_finds_the_planted_segment_first_and_at_the_right_time():
    cache = make_cache({0: (*background(), 0.0, 120.0), 1: (*background(), 0.0, 120.0),
                        2: plant(build_shape(), at_s=40), 3: plant(fade_shape(), at_s=40)})
    response = search(cache, Query.from_dict(BUILD_QUERY))
    best = response.results[0]
    assert best.track_id == 2 and abs(best.start_s - 40) <= 1 and best.end_s - best.start_s == pytest.approx(30, abs=1)
    assert best.overall_score > 90
    runner_up = response.results[1]
    assert best.overall_score - runner_up.overall_score > 40    # a clear winner: the fade and flat tracks do not fool it


def test_finds_a_time_stretched_copy_by_trying_other_lengths():
    cache = make_cache({0: (*background(), 0.0, 120.0), 1: plant(build_shape(seconds=24), at_s=50)})
    best = search(cache, Query.from_dict(BUILD_QUERY)).results[0]    # a 30 s sketch, but the music is 24 s
    assert best.track_id == 1 and abs(best.start_s - 50) <= 1
    assert best.end_s - best.start_s == pytest.approx(24, abs=0.5) and best.overall_score > 80


def test_a_flat_sketch_only_matches_flat_music():
    # tracks 0 and 1 are usable only where their shape is (30-60 s); track 2 is flat throughout
    build = (*plant(build_shape(), at_s=30)[:2], 30.0, 60.0)
    fade = (*plant(fade_shape(), at_s=30)[:2], 30.0, 60.0)
    cache = make_cache({0: build, 1: fade, 2: (*background(), 0.0, 120.0)})
    flat = {"features": {"loudness": {"points": [[0, 0.5], [30, 0.5]]}}, "duration_s": 30, "top_k": 6}
    results = search(cache, Query.from_dict(flat)).results
    assert results[0].track_id == 2 and results[0].overall_score > 60          # the flat track wins
    shaped = [r for r in results if r.track_id in (0, 1)]
    assert shaped and all(r.overall_score < 20 for r in shaped)                # shaped music scores low


def test_zero_weight_ignores_a_feature_entirely():
    good = plant(build_shape(), 40)
    wrong_brightness = (good[0], np.full(N, 9.0) + RNG.normal(0, 0.02, N), 0.0, 120.0)
    cache = make_cache({0: good, 1: wrong_brightness})
    both = search(cache, Query.from_dict(BUILD_QUERY))
    assert [r.track_id for r in both.results[:2]] == [0, 1] and both.results[0].overall_score > both.results[1].overall_score + 20

    only_loudness = dict(BUILD_QUERY, features={**BUILD_QUERY["features"],
                                                "brightness": {"points": [[0, 0.1], [30, 0.9]], "weight": 0}})
    response = search(cache, Query.from_dict(only_loudness))
    assert set(response.results[0].feature_scores) == {"loudness"}
    assert response.weights == {"loudness": 1.0, "brightness": 0.0}
    assert {r.track_id for r in response.results[:2]} == {0, 1}
    assert abs(response.results[0].overall_score - response.results[1].overall_score) < 5


def test_default_weights_are_equal_and_are_always_returned():
    cache = make_cache({0: plant(build_shape(), 40)})
    assert search(cache, Query.from_dict(BUILD_QUERY)).weights == {"loudness": 0.5, "brightness": 0.5}


def test_music_in_the_silent_intro_is_never_returned():
    loud, bright, _, _ = plant(build_shape(), at_s=10)           # a perfect match at 10-40 s ...
    cache = make_cache({0: (loud, bright, 70.0, 120.0)})         # ... but the usable audio starts at 70 s
    results = search(cache, Query.from_dict(BUILD_QUERY)).results
    assert results and all(r.start_s >= 70 for r in results)


def test_results_are_best_first_limited_and_json_ready():
    cache = make_cache({t: plant(build_shape(), 30 + t) for t in range(6)})
    response = search(cache, Query.from_dict(dict(BUILD_QUERY, top_k=4)))
    scores = [r.overall_score for r in response.results]
    assert len(scores) == 4 and scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 100 for s in scores) and response.search_time_s > 0
    import json
    json.dumps(response.to_dict())


def test_unknown_feature_is_a_clear_error():
    cache = make_cache({0: plant(build_shape(), 40)})
    with pytest.raises(QueryError, match=r"unknown feature.*'warmth'.*available features: \['loudness', 'brightness'\]"):
        search(cache, Query.from_dict({"features": {"warmth": {"points": [[0, 0], [30, 1]]}}, "duration_s": 30}))


# ---- picking the top results (task 4.11) ----

def test_overlap_is_measured_against_the_shorter_stretch():
    assert overlap_fraction(0, 30, 15, 45) == pytest.approx(0.5)
    assert overlap_fraction(0, 30, 40, 70) == 0.0
    assert overlap_fraction(0, 37.5, 5, 29) == 1.0           # the short one sits fully inside the long one


def candidates():
    # (track, start, end), already sorted best first
    rows = [(1, 10, 40), (1, 11, 41), (1, 30, 60), (1, 60, 90), (2, 10, 40), (2, 12, 42), (3, 0, 30)]
    track, s, e = (np.array(c, dtype=float) for c in zip(*rows))
    return np.arange(len(rows)), track, s, e


def test_overlapping_and_excess_results_from_one_track_are_dropped():
    order, track, s, e = candidates()
    kept = select_top(order, track, s, e, top_k=10)
    # (1,11-41) overlaps the first by 97% -> out; (1,30-60) overlaps it by 33% -> kept; (1,60-90) is a 3rd from track 1 -> out
    # (2,12-42) overlaps (2,10-40) by 93% -> out
    assert kept == [0, 2, 4, 6]


def test_exactly_half_overlap_is_allowed_and_top_k_stops_the_walk():
    order = np.arange(2)
    track, s, e = np.array([1.0, 1.0]), np.array([0.0, 15.0]), np.array([30.0, 45.0])
    assert select_top(order, track, s, e, top_k=5) == [0, 1]
    assert select_top(order, track, s, e, top_k=1) == [0]