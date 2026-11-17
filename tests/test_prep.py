import numpy as np
import pytest

from shapesearch.db import add_track, connect, create_schema, register_feature, save_curve, save_hits
from shapesearch.features.smoothing import smooth
from shapesearch.matching.prep import (CurveCache, DEFAULT_MIN_RANGE, normalize, normalize_feature,
                                       normalize_sketch, thin)


def test_thin_averages_groups_of_five_and_drops_the_leftover():
    curve = np.arange(13, dtype=float)                # 0..12 -> groups [0-4], [5-9], leftover [10-12]
    assert list(thin(curve)) == [2.0, 7.0]
    assert thin(np.zeros(600)).shape == (120,)        # 60 s at 10 per second -> 120 values


@pytest.fixture
def con(tmp_path):
    con = connect(tmp_path / "t.db")
    create_schema(con)
    register_feature(con, "loudness", "loudness_v1", "LUFS", "d", "m", {})
    register_feature(con, "brightness", "brightness_v1", "octaves", "d", "m", {})
    rng = np.random.default_rng(0)
    for tid in (0, 1):
        n = 600 + 7 * tid                              # track 1 has a leftover partial group
        add_track(con, tid, f"Track {tid}", "x.wav", n / 10, 2.0, n / 10 - 1.0)
        save_curve(con, tid, "loudness", -20 + rng.normal(0, 3, n))
        save_curve(con, tid, "brightness", 9 + rng.normal(0, 0.3, n))
        save_hits(con, tid, "loudness", [(12.0 + tid, 8.0)])
    yield con
    con.close()


def test_cache_holds_smoothed_thinned_curves(con):
    cache = CurveCache.from_database(con)
    assert len(cache) == 2 and cache.track_ids == [0, 1]
    assert cache.feature_names == ["loudness", "brightness"]
    assert cache.curve(0, "loudness").shape == (120,)           # 600 values -> 120 groups of 5
    assert cache.curve(1, "loudness").shape == (121,)           # 607 values -> 121 groups, 2 left over
    assert cache.curve(0, "loudness").dtype == np.float32


def test_cache_values_are_smooth_then_thin(con):
    from shapesearch.db import load_curve
    cache = CurveCache.from_database(con, smooth_s=1.5)
    expected = thin(smooth(load_curve(con, 0, "brightness"), 1.5))
    assert np.array_equal(cache.curve(0, "brightness"), expected)
    raw_jitter = np.std(np.diff(thin(load_curve(con, 0, "brightness"))))
    assert np.std(np.diff(cache.curve(0, "brightness"))) < raw_jitter    # smoothing really calmed it


def test_cache_keeps_usable_range_and_hits(con):
    cache = CurveCache.from_database(con)
    assert cache.track_info(0) == {"start_s": 2.0, "end_s": 59.0, "duration_s": 60.0}
    assert cache.hits(1) == [(13.0, 8.0)]
    assert cache.settings["smooth_s"] == 1.5 and cache.nbytes == (120 + 121) * 2 * 4


def test_cache_picks_up_a_new_feature_without_code_changes(con):
    register_feature(con, "drum_loudness", "v1", "LUFS", "d", "m", {})
    for tid in (0, 1):
        save_curve(con, tid, "drum_loudness", np.linspace(-30, -10, 600 + 7 * tid))
    cache = CurveCache.from_database(con)
    assert "drum_loudness" in cache.feature_names
    assert cache.curve(0, "drum_loudness").shape == (120,)
    assert cache.curve(1, "drum_loudness").shape == (121,)


def test_normalize_rescales_to_zero_one():
    out = normalize(np.array([-30.0, -20.0, -10.0]), min_range=3.0)
    assert list(out) == [0.0, 0.5, 1.0]


def test_flatness_guard_keeps_a_nearly_flat_curve_small():
    flat = np.array([-20.0, -19.5, -19.0, -19.8, -20.0])          # varies by only 1 LU
    out = normalize(flat, min_range=3.0)
    assert out.min() == 0.0 and out.max() == pytest.approx(1.0 / 3.0)   # not stretched to 1
    unguarded = (flat - flat.min()) / (flat.max() - flat.min())
    assert unguarded.max() == 1.0                                  # what it would have been


def test_a_perfectly_constant_curve_normalizes_to_zeros_without_errors():
    out = normalize(np.full(10, -14.0), min_range=3.0)
    assert np.all(out == 0.0) and np.all(np.isfinite(out))


def test_normalize_works_on_a_stack_of_windows_row_by_row():
    rng = np.random.default_rng(1)
    stack = rng.normal(-20, 4, (5, 60))
    together = normalize(stack, 3.0)
    assert together.shape == (5, 60)
    for i in range(5):
        assert np.allclose(together[i], normalize(stack[i], 3.0))


def test_each_feature_has_its_own_guard_and_unknown_ones_get_a_default():
    wiggle = np.array([0.0, 0.2, 0.0, 0.2])
    assert normalize_feature(wiggle, "brightness").max() == pytest.approx(0.2 / 0.3)   # 0.3 octave guard
    assert normalize_feature(wiggle, "loudness").max() == pytest.approx(0.2 / 3.0)
    assert normalize_feature(wiggle, "drum_loudness").max() == pytest.approx(0.2 / DEFAULT_MIN_RANGE)


def test_sketch_normalization_stretches_a_real_shape_but_not_a_near_flat_line():
    assert list(normalize_sketch(np.array([0.3, 0.45, 0.6]))) == pytest.approx([0.0, 0.5, 1.0])   # range 0.3 > 0.2
    assert normalize_sketch(np.array([0.5, 0.55, 0.5])).max() == pytest.approx(0.25)              # range 0.05, guarded
    assert np.all(normalize_sketch(np.full(8, 0.5)) == 0.0)