import csv
from pathlib import Path

from shapesearch.db import add_track, save_curve, save_hits
from shapesearch.features.extract import OUT_DIR, MANIFEST, features_path, load_features, status
from shapesearch.features.registry import register_all


def _blank_to_none(value):
    value = (value or "").strip()
    return value or None


def load_catalog(con, manifest=MANIFEST, features_dir=OUT_DIR):
    """Register the features, then load every track in the manifest.

    Safe to run repeatedly: tracks and curves are replaced, never duplicated.
    A track whose features are missing or out of date is skipped and reported;
    it never stops the others. Returns {'loaded': n, 'failed': [(track_id, why)]}.
    """
    manifest = Path(manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found. Run this from the project root.")
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    register_all(con)
    loaded, failed = 0, []
    for row in rows:
        track_id = int(row["track_id"])
        try:
            state = status(features_path(track_id, features_dir))
            if state == "missing":
                raise RuntimeError("no extracted features; run: python -m shapesearch.features.extract")
            if state == "stale":
                raise RuntimeError("extracted features are out of date; run: python -m shapesearch.features.extract")
            f = load_features(track_id, features_dir)

            add_track(
                con, track_id,
                title=_blank_to_none(row.get("title")) or f"Track {track_id}",
                file_path=row["file_path"],
                duration_s=f["duration_s"],       # exact length, not the manifest's rounded one
                start_s=f["start_s"], end_s=f["end_s"],
                artist=_blank_to_none(row.get("artist")), genre=_blank_to_none(row.get("genre")),
                source=_blank_to_none(row.get("source")), source_url=_blank_to_none(row.get("source_url")),
                license=_blank_to_none(row.get("license")),
            )
            save_curve(con, track_id, "loudness", f["loudness"], f["hop_s"])
            save_curve(con, track_id, "brightness", f["brightness"], f["hop_s"])
            save_hits(con, track_id, "loudness", f["hits"])
            loaded += 1
        except Exception as exc:
            failed.append((track_id, str(exc)))
    return {"loaded": loaded, "failed": failed}