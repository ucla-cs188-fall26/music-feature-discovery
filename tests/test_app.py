import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from shapesearch.app import create_app
from shapesearch.db import add_track, connect, create_schema, register_feature, save_curve, save_hits
from synthetic import background, build_shape, fade_shape

BUILD_QUERY = {"features": {"loudness": {"points": [[0, 0.1], [20, 0.9], [30, 0.9]]},
                            "brightness": {"points": [[0, 0.1], [30, 0.9]]}}, "duration_s": 30, "top_k": 5}


def per_tenth(curve_per_half_second):
    """The stored curves have 10 values per second; the test shapes have 2."""
    return np.repeat(curve_per_half_second, 5)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo")
    wav = folder / "clip.wav"
    sf.write(wav, np.sin(np.linspace(0, 2000, 20 * 44100)).astype(np.float32).repeat(2).reshape(-1, 2) * 0.3, 44100)

    con = connect(folder / "demo.db")
    create_schema(con)
    register_feature(con, "loudness", "loudness_v1", "LUFS", "How loud it sounds", "short-term", {})
    register_feature(con, "brightness", "brightness_v1", "octaves", "How bright it sounds", "centroid", {})
    rng = np.random.default_rng(0)
    shapes = {0: None, 1: fade_shape(), 2: build_shape()}
    for tid, shape in shapes.items():
        loud, bright = background(rng)
        if shape is not None:
            loud[80:80 + 60], bright[80:80 + 60] = shape                  # planted at 40 s
        add_track(con, tid, f"Song {tid}", str(wav), 120.0, 0.0, 120.0)
        save_curve(con, tid, "loudness", per_tenth(loud))
        save_curve(con, tid, "brightness", per_tenth(bright))
        save_hits(con, tid, "loudness", [(48.0, 7.0)] if tid == 2 else [])
    con.close()

    with TestClient(create_app(folder / "demo.db")) as client:
        yield client


def test_the_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "Shape Search" in response.text


def test_info_lists_features_and_tracks(client):
    info = client.get("/api/info").json()
    assert info["tracks"] == 3
    assert [f["name"] for f in info["features"]] == ["loudness", "brightness"]


@pytest.mark.parametrize("method", ["dtw", "euclidean"])
def test_search_returns_everything_the_page_draws(client, method):
    data = client.post("/api/search", json={"query": BUILD_QUERY, "method": method}).json()
    best = data["results"][0]
    assert best["track_id"] == 2 and best["title"] == "Song 2"
    assert 38 <= best["start_s"] and best["end_s"] <= 71                  # inside the planted stretch (40-70 s)
    assert data["method"] == method and data["duration_s"] == 30 and data["search_time_s"] > 0
    assert set(data["sketch"]) == set(best["curves"]) == {"loudness", "brightness"}
    assert len(data["sketch"]["loudness"]) == 60
    assert all(0 <= v <= 1 for v in best["curves"]["loudness"])
    assert abs(len(best["curves"]["loudness"]) - (best["end_s"] - best["start_s"]) * 2) <= 1
    assert (len(best["path"]) > 0) == (method == "dtw")


def test_anchored_search_reports_the_hit(client):
    query = dict(BUILD_QUERY, mode="anchored", anchors_s=[8])
    data = client.post("/api/search", json={"query": query, "method": "dtw"}).json()
    assert data["mode"] == "anchored" and [r["track_id"] for r in data["results"]] == [2]
    assert data["results"][0]["anchors"][0]["hit_time_s"] == 48.0


@pytest.mark.parametrize("body, message", [
    ({"query": {"features": {"warmth": {"points": [[0, 0], [30, 1]]}}, "duration_s": 30}}, "unknown feature"),
    ({"query": {"features": {"loudness": {"points": [[0, 0]]}}, "duration_s": 30}}, "at least 2 points"),
    ({"query": BUILD_QUERY, "method": "magic"}, "method must be one of"),
    ({}, "missing required field"),
])
def test_a_bad_search_is_a_400_with_a_readable_message(client, body, message):
    response = client.post("/api/search", json=body)
    assert response.status_code == 400 and message in response.json()["detail"]


def test_audio_returns_just_the_requested_stretch(client):
    response = client.get("/api/audio/2", params={"start": 2, "end": 7})
    assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
    data, rate = sf.read(io.BytesIO(response.content))
    assert rate == 22050 and data.ndim == 1 and len(data) / rate == pytest.approx(5.0, abs=0.01)


def test_audio_errors_are_clear(client):
    assert client.get("/api/audio/99").status_code == 404
    assert client.get("/api/audio/2", params={"start": 5, "end": 5}).status_code == 400
    assert client.get("/api/audio/2", params={"start": 30, "end": 40}).status_code == 400    # the clip is only 20 s long


def test_a_missing_database_stops_startup_with_a_helpful_message(tmp_path):
    with pytest.raises(RuntimeError, match="Database not found"):
        with TestClient(create_app(tmp_path / "nope.db")):
            pass