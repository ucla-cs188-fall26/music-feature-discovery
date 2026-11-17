import numpy as np
import pytest

from shapesearch.matching.anchors import guide_knots, match_anchors


def test_a_window_passes_when_a_hit_lands_where_the_anchor_says():
    hits = [(48.0, 7.0)]
    starts = np.array([38.0, 39.0, 40.0, 41.0])
    ok, hit_t, strength = match_anchors(hits, [8.0], 30.0, starts, np.full(4, 30.0))
    assert ok.tolist() == [False, False, True, False]        # only a window starting at 40 puts 8 s at 48
    assert hit_t[2, 0] == 48.0 and strength[2, 0] == 7.0


def test_the_tolerance_is_half_a_second_either_side():
    hits = [(48.0, 7.0)]
    starts = np.array([39.4, 39.5, 40.5, 40.6])
    ok, _, _ = match_anchors(hits, [8.0], 30.0, starts, np.full(4, 30.0))
    assert ok.tolist() == [False, True, True, False]


def test_a_longer_window_expects_the_hit_proportionally_later():
    hits = [(60.0, 5.0)]
    # a 37.5 s window (1.25x the 30 s sketch): an anchor at 8 s is expected 10 s in, so start at 50
    ok, _, _ = match_anchors(hits, [8.0], 30.0, np.array([50.0, 52.0]), np.full(2, 37.5))
    assert ok.tolist() == [True, False]


def test_no_hits_means_no_window_passes_and_there_is_no_error():
    ok, _, _ = match_anchors([], [8.0], 30.0, np.array([40.0, 41.0]), np.full(2, 30.0))
    assert not ok.any()


def test_hit_order_does_not_matter_but_each_anchor_needs_its_own_hit_in_order():
    hits = [(58.0, 4.0), (48.0, 7.0)]                         # stored out of order
    ok, hit_t, _ = match_anchors(hits, [8.0, 18.0], 30.0, np.array([40.0]), np.full(1, 30.0))
    assert ok.tolist() == [True] and hit_t[0].tolist() == [48.0, 58.0]

    one_hit = [(48.0, 7.0)]                                   # two close anchors cannot share a single hit
    ok, _, _ = match_anchors(one_hit, [8.0, 8.4], 30.0, np.array([40.0]), np.full(1, 30.0))
    assert not ok.any()


def test_guide_knots_pair_each_anchor_with_its_hit_cell():
    # a 30 s sketch is 60 cells; the anchor at 8 s is cell 16. Window starts at 40 s; the hit at 48.6 s is window cell 17.
    knots_i, knots_j = guide_knots([8.0], np.array([[48.6]]), np.array([40.0]), m=60, n=66)
    assert knots_i.tolist() == [0, 16, 59]
    assert knots_j.tolist() == [[0.0, 17.0, 65.0]]


def test_guide_knots_ignore_anchors_at_the_very_ends_and_duplicates():
    knots_i, knots_j = guide_knots([0.0, 8.0, 8.2, 30.0], np.array([[40.0, 48.0, 48.2, 70.0]]),
                                   np.array([40.0]), m=60, n=60)
    assert knots_i.tolist() == [0, 16, 59]                    # the ends are already fixed; 8.0 and 8.2 share cell 16
    assert knots_j.shape == (1, 3)


def test_guide_knots_without_usable_anchors_is_the_plain_diagonal():
    knots_i, knots_j = guide_knots([0.0], np.array([[40.0]]), np.array([40.0]), m=60, n=66)
    assert knots_i.tolist() == [0, 59] and knots_j.tolist() == [[0.0, 65.0]]