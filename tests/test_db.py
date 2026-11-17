import numpy as np
import pytest

from shapesearch.db import (add_track, connect, create_schema, get_track, list_features,
                            list_tracks, load_curve, load_hits, register_feature,
                            save_curve, save_hits, table_counts)


@pytest.fixture
def con(tmp_path):
    con = connect(tmp_path / "test.db")
    create_schema(con)
    register_feature(con, "loudness", "loudness_v1", "LUFS", "How loud it sounds",
                     "ITU-R BS.1770 short-term, 3 s window", {"window_s": 3.0})
    add_track(con, 0, "Rise", "data/processed/0000.wav", 180.0, 0.0, 180.0, genre="Electronic")
    yield con
    con.close()


def test_curve_round_trip_is_identical(con):
    rng = np.random.default_rng(0)
    curve = rng.uniform(-70, -5, 1800).astype(np.float32)
    save_curve(con, 0, "loudness", curve)
    loaded = load_curve(con, 0, "loudness")
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, curve)          # bit-for-bit the same numbers


def test_float64_input_is_stored_as_float32(con):
    save_curve(con, 0, "loudness", np.array([-20.123456789, -10.5]))
    assert np.array_equal(load_curve(con, 0, "loudness"), np.array([-20.123456789, -10.5], dtype=np.float32))


def test_loaded_curve_is_writable(con):
    save_curve(con, 0, "loudness", [1.0, 2.0, 3.0])
    load_curve(con, 0, "loudness")[0] = 99.0       # must not raise


def test_saving_again_replaces_the_curve(con):
    save_curve(con, 0, "loudness", [1.0, 2.0, 3.0])
    save_curve(con, 0, "loudness", [4.0, 5.0])
    assert list(load_curve(con, 0, "loudness")) == [4.0, 5.0]
    assert table_counts(con)["curves"] == 1


@pytest.mark.parametrize("bad", [[1.0, float("nan")], [1.0, float("inf")], [], [[1.0, 2.0]]])
def test_bad_curves_are_rejected(con, bad):
    with pytest.raises(ValueError):
        save_curve(con, 0, "loudness", bad)


def test_unknown_names_give_helpful_errors(con):
    with pytest.raises(KeyError, match="registered features"):
        save_curve(con, 0, "no_such_feature", [1.0])
    with pytest.raises(KeyError, match="unknown track 42"):
        save_curve(con, 42, "loudness", [1.0])
    with pytest.raises(KeyError, match="no 'loudness' curve"):
        load_curve(con, 0, "loudness")


def test_tracks_round_trip_update_and_filter_by_genre(con):
    add_track(con, 1, "Calm", "data/processed/0001.wav", 240.0, 3.0, 238.0, genre="Ambient")
    add_track(con, 0, "Rise (renamed)", "data/processed/0000.wav", 180.0, 0.0, 180.0, genre="Electronic")
    assert len(list_tracks(con)) == 2                       # updated, not duplicated
    assert get_track(con, 0)["title"] == "Rise (renamed)"
    assert [t["title"] for t in list_tracks(con, genre="Ambient")] == ["Calm"]
    assert get_track(con, 99) is None
    assert get_track(con, 1)["license"] is None             # license is optional


def test_register_feature_keeps_its_id_and_parses_params(con):
    first = register_feature(con, "loudness", "loudness_v2", "LUFS", "How loud", "new method", {"window_s": 2.0})
    assert first == 1                                       # same id as before
    (feature,) = list_features(con)
    assert feature["version"] == "loudness_v2" and feature["params"] == {"window_s": 2.0}


def test_hits_round_trip_and_replace(con):
    save_hits(con, 0, "loudness", [(64.5, 12.0), (20.1, 8.5)])
    assert load_hits(con, 0) == [(20.1, 8.5), (64.5, 12.0)]  # earliest first
    save_hits(con, 0, "loudness", [(30.0, 5.0)])
    assert load_hits(con, 0) == [(30.0, 5.0)]
    save_hits(con, 0, "loudness", [])
    assert load_hits(con, 0) == []


def test_a_brand_new_feature_needs_no_code_changes(con):
    """The point of the registry: a third feature is just another row."""
    register_feature(con, "drum_loudness", "drum_loudness_v1", "LUFS",
                     "Loudness of the drum stem", "BS.1770 on the drum stem", {})
    save_curve(con, 0, "drum_loudness", [-30.0, -20.0, -25.0])
    save_hits(con, 0, "drum_loudness", [(1.0, 9.0)])
    assert [f["name"] for f in list_features(con)] == ["loudness", "drum_loudness"]
    assert list(load_curve(con, 0, "drum_loudness")) == [-30.0, -20.0, -25.0]
    assert load_hits(con, 0, "drum_loudness") == [(1.0, 9.0)]
    assert load_hits(con, 0, "loudness") == []               # features do not mix