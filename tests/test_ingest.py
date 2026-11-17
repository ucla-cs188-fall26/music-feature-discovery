import numpy as np
import pytest
import soundfile as sf

from shapesearch import ingest
from shapesearch.db import (connect, create_schema, list_features, list_tracks, load_curve,
                            load_hits, save_curve, table_counts)
from shapesearch.features import extract
from shapesearch.integrity import find_problems

SR = 44100


@pytest.fixture
def catalog(tmp_path):
    """Three short synthetic tracks, extracted for real, plus a database to load them into."""
    rng = np.random.default_rng(0)
    lines = ["track_id,title,artist,source,source_url,license,file_path,duration_s,genre"]
    for tid in range(3):
        n = int(SR * 20.07)                                   # a length that is not a whole number of steps
        amp = np.where(np.arange(n) / SR < 10, 0.02, 0.2)     # a drop at 10 s
        x = (amp * rng.standard_normal(n)).astype(np.float32)
        path = tmp_path / f"{tid:04d}.wav"
        sf.write(path, np.stack([x, x], axis=1), SR)
        genre = "Electronic" if tid < 2 else ""
        lines.append(f"{tid},Track {tid},,,,,{path},20.07,{genre}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n")
    features_dir = tmp_path / "features"
    extract.run(manifest, features_dir, tmp_path / "fail.log")
    con = connect(tmp_path / "shapes.db")
    create_schema(con)
    yield dict(con=con, manifest=manifest, features=features_dir, tmp=tmp_path)
    con.close()


def load(c):
    return ingest.load_catalog(c["con"], c["manifest"], c["features"])


def test_loads_every_track_and_registers_the_features(catalog):
    result = load(catalog)
    con = catalog["con"]
    assert result == {"loaded": 3, "failed": []}
    counts = table_counts(con)
    assert counts["tracks"] == 3 and counts["features"] == 2 and counts["curves"] == 6
    names = [f["name"] for f in list_features(con)]
    assert names == ["loudness", "brightness"]
    loudness = list_features(con)[0]
    assert loudness["version"] == "loudness_v1" and loudness["unit"] == "LUFS"
    assert loudness["params"]["window_s"] == 3.0 and "hit_detection" in loudness["params"]


def test_database_holds_exactly_what_was_extracted(catalog):
    load(catalog)
    saved = extract.load_features(1, catalog["features"])
    assert np.array_equal(load_curve(catalog["con"], 1, "loudness"), saved["loudness"])
    assert np.array_equal(load_curve(catalog["con"], 1, "brightness"), saved["brightness"])
    assert load_hits(catalog["con"], 1) == saved["hits"]
    track = {t["track_id"]: t for t in list_tracks(catalog["con"])}[1]
    assert track["duration_s"] == pytest.approx(saved["duration_s"], abs=1e-9)   # exact, not rounded
    assert track["start_s"] == saved["start_s"] and track["end_s"] == saved["end_s"]


def test_blank_manifest_fields_become_null(catalog):
    load(catalog)
    tracks = {t["track_id"]: t for t in list_tracks(catalog["con"])}
    assert tracks[0]["genre"] == "Electronic" and tracks[2]["genre"] is None
    assert tracks[0]["license"] is None and tracks[0]["title"] == "Track 0"


def test_loading_twice_does_not_duplicate_anything(catalog):
    load(catalog)
    first = table_counts(catalog["con"])
    load(catalog)
    assert table_counts(catalog["con"]) == first


def test_track_without_features_is_reported_and_others_still_load(catalog):
    extract.features_path(2, catalog["features"]).unlink()
    result = load(catalog)
    assert result["loaded"] == 2
    assert [t for t, _ in result["failed"]] == [2]
    assert "shapesearch.features.extract" in result["failed"][0][1]


def test_out_of_date_features_are_refused(catalog, monkeypatch):
    monkeypatch.setitem(extract.PARAMS["hits"], "min_rise_lu", 6.0)
    result = load(catalog)
    assert result["loaded"] == 0 and len(result["failed"]) == 3
    assert "out of date" in result["failed"][0][1]


def test_a_freshly_loaded_catalog_passes_every_check(catalog):
    load(catalog)
    assert find_problems(catalog["con"]) == []


def test_check_finds_a_track_missing_a_feature(catalog):
    load(catalog)
    catalog["con"].execute("DELETE FROM curves WHERE track_id = 1 AND feature_id = 2")
    assert find_problems(catalog["con"]) == ["track 1: missing the brightness curve"]


def test_check_finds_a_curve_of_the_wrong_length(catalog):
    load(catalog)
    save_curve(catalog["con"], 0, "loudness", np.zeros(50))
    (problem,) = find_problems(catalog["con"])
    assert "track 0 loudness" in problem and "has 50 values" in problem


def test_check_finds_missing_values_hidden_in_a_curve(catalog):
    load(catalog)
    con = catalog["con"]
    n = con.execute("SELECT n_values FROM curves WHERE track_id = 2 AND feature_id = 1").fetchone()[0]
    bad = np.full(n, np.nan, dtype="<f4").tobytes()      # save_curve refuses NaN, so go around it
    con.execute("UPDATE curves SET values_blob = ? WHERE track_id = 2 AND feature_id = 1", (bad,))
    assert find_problems(con) == ["track 2 loudness: contains missing or infinite values"]


def test_check_finds_a_hit_outside_the_track(catalog):
    load(catalog)
    catalog["con"].execute("INSERT INTO hit_points (track_id, time_s, strength, feature_id) VALUES (0, 999.0, 5.0, 1)")
    (problem,) = find_problems(catalog["con"])
    assert "outside the track" in problem