"""A small web app for the demo: serves the sketch page and answers searches.

Run from the project root:
    uvicorn shapesearch.app:app
then open http://127.0.0.1:8000 in a browser.

    GET  /                    the page (shapesearch/static/index.html)
    GET  /api/info            the features and how many tracks are loaded
    POST /api/search          {"query": {...}, "method": "dtw" | "euclidean"} -> results
    GET  /api/audio/{id}      ?start=..&end=.. a short WAV of that stretch of a track
"""
import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from scipy.signal import resample_poly

from shapesearch.db import DB_PATH, connect, list_features, list_tracks
from shapesearch.matching.prep import CurveCache, normalize_feature, normalize_sketch
from shapesearch.matching.query import SKETCH_RATE, Query, query_curves
from shapesearch.matching.search import DEFAULT_METHOD, search

STATIC_DIR = Path(__file__).parent / "static"
MAX_CLIP_S = 180      # the longest stretch of audio the audio endpoint will cut


def _rounded(values):
    return np.round(np.asarray(values, dtype=np.float64), 3).tolist()


def create_app(db_path=DB_PATH):
    """Build the app. The database is read once at startup and kept in memory."""

    @asynccontextmanager
    async def lifespan(app):
        if not Path(db_path).exists():
            raise RuntimeError(f"Database not found at {db_path}. Run this from the project root, "
                               f"after python scripts/build_db.py --load")
        con = connect(db_path)
        try:
            app.state.cache = CurveCache.from_database(con)
            app.state.tracks = {t["track_id"]: t for t in list_tracks(con)}
            app.state.features = [{"name": f["name"], "unit": f["unit"], "description": f["description"]}
                                  for f in list_features(con)]
        finally:
            con.close()
        yield

    app = FastAPI(title="Shape Search demo", lifespan=lifespan)

    @app.get("/")
    def page():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/info")
    def info():
        return {"tracks": len(app.state.tracks), "features": app.state.features, "default_method": DEFAULT_METHOD}

    @app.post("/api/search")
    def run_search(body: dict = Body(...)):
        cache, tracks = app.state.cache, app.state.tracks
        method = body.get("method", DEFAULT_METHOD)
        try:
            query = Query.from_dict(body.get("query") or {}, known_features=cache.feature_names)
            response = search(cache, query, method=method)
        except ValueError as exc:                    # a bad query or an unknown method: tell the page what to fix
            raise HTTPException(status_code=400, detail=str(exc)) from None

        out = response.to_dict()
        out["method"] = method
        out["duration_s"] = query.duration_s
        out["sketch"] = {f: _rounded(normalize_sketch(c)) for f, c in query_curves(query).items()}
        for result, entry in zip(response.results, out["results"]):
            track = tracks[result.track_id]
            entry["title"] = track["title"]
            lo, hi = round(result.start_s * SKETCH_RATE), round(result.end_s * SKETCH_RATE)
            # the matched stretch, rescaled the way the search saw it, so the page can draw it over the sketch
            entry["curves"] = {f: _rounded(normalize_feature(cache.curve(result.track_id, f)[lo:hi], f))
                               for f in out["sketch"]}
        return out

    @app.get("/api/audio/{track_id}")
    def audio(track_id: int, start: float = 0.0, end: float | None = None):
        track = app.state.tracks.get(track_id)
        if track is None:
            raise HTTPException(status_code=404, detail=f"no track {track_id}")
        path = Path(track["file_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"audio file {path} not found (run from the project root)")
        with sf.SoundFile(path) as f:
            rate = f.samplerate
            first = max(0, int(start * rate))
            last = f.frames if end is None else min(f.frames, int(end * rate))
            if last <= first or (last - first) / rate > MAX_CLIP_S:
                raise HTTPException(status_code=400, detail=f"start/end must select between 0 and {MAX_CLIP_S} seconds")
            f.seek(first)
            data = f.read(last - first, dtype="float32")
        mono = data.mean(axis=1) if data.ndim > 1 else data
        if rate % 2 == 0:                            # half the sample rate: a smaller download, fine for listening
            mono, rate = resample_poly(mono, 1, 2), rate // 2
        buffer = io.BytesIO()
        sf.write(buffer, mono, rate, format="WAV", subtype="PCM_16")
        return Response(buffer.getvalue(), media_type="audio/wav")

    return app


app = create_app()