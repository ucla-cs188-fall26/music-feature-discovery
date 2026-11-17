import numpy as np
import pytest
import soundfile as sf

from shapesearch.audio import time_grid
from shapesearch.features import extract

SR = 44100


@pytest.fixture
def catalog(tmp_path):
    """Three short synthetic tracks plus a manifest pointing at them."""
    rng = np.random.default_rng(0)
    lines = ["track_id,title,artist,source,source_url,license,file_path,duration_s,genre"]
    for tid in range(3):
        n = SR * 20
        amp = np.where(np.arange(n) / SR < 10, 0.02, 0.2)   # a drop at 10 s
        x = (amp * rng.standard_normal(n)).astype(np.float32)
        path = tmp_path / f"{tid:04d}.wav"
        sf.write(path, np.stack([x, x], axis=1), SR)
        lines.append(f"{tid},Track {tid},,,,,{path},20.0,")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n")
    return dict(manifest=manifest, out_dir=tmp_path / "features", failure_log=tmp_path / "fail.log",
                tmp=tmp_path, lines=lines)


def run(catalog, **kw):
    return extract.run(catalog["manifest"], catalog["out_dir"], catalog["failure_log"], **kw)


def test_processes_every_track_and_saves_usable_output(catalog):
    result = run(catalog)
    assert result["done"] == 3 and result["failed"] == []
    f = extract.load_features(1, catalog["out_dir"])
    n = len(time_grid(20.0))
    assert len(f["loudness"]) == len(f["brightness"]) == n
    assert len(f["hits"]) == 1 and abs(f["hits"][0][0] - 10.0) <= 0.3
    assert f["meta"]["versions"]["loudness"] == "loudness_v1"
    assert f["hop_s"] == 0.1 and f["duration_s"] == pytest.approx(20.0)


def test_second_run_skips_finished_tracks(catalog):
    run(catalog)
    result = run(catalog)
    assert result["done"] == 0 and result["skipped"] == 3


def test_changed_settings_trigger_recompute(catalog, monkeypatch):
    run(catalog)
    monkeypatch.setitem(extract.PARAMS["hits"], "min_rise_lu", 6.0)
    assert run(catalog)["done"] == 3


def test_bumped_version_triggers_recompute(catalog, monkeypatch):
    run(catalog)
    monkeypatch.setitem(extract.FEATURE_VERSIONS, "loudness", "loudness_v2")
    assert run(catalog)["done"] == 3


def test_force_redoes_everything(catalog):
    run(catalog)
    assert run(catalog, force=True)["done"] == 3


def test_only_limits_which_tracks_run(catalog):
    result = run(catalog, only=[2])
    assert result["done"] == 1
    assert extract.features_path(2, catalog["out_dir"]).exists()
    assert not extract.features_path(0, catalog["out_dir"]).exists()


def test_bad_track_is_logged_and_the_rest_still_run(catalog):
    lines = catalog["lines"] + [f"3,Missing,,,,,{catalog['tmp'] / 'nope.wav'},20.0,"]
    catalog["manifest"].write_text("\n".join(lines) + "\n")
    result = run(catalog)
    assert result["done"] == 3
    assert [tid for tid, _ in result["failed"]] == [3]
    assert "track 3" in catalog["failure_log"].read_text()


def test_corrupt_output_file_is_redone(catalog):
    run(catalog)
    extract.features_path(0, catalog["out_dir"]).write_bytes(b"not a real file")
    assert run(catalog)["done"] == 1