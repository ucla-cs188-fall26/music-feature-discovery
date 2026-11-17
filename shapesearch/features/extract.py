import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from shapesearch.audio import HOP_S, load_audio, time_grid
from shapesearch.features.brightness import N_FFT, STFT_HOP, brightness_curve
from shapesearch.features.hits import detect_hits
from shapesearch.features.loudness import loudness_curve, usable_range

MANIFEST = Path("data/manifest.csv")
OUT_DIR = Path("data/features")
FAILURE_LOG = Path("data/extraction_failures.log")

# If you change what a feature MEANS, bump its version (e.g. loudness_v2).
# If you only change a setting, the PARAMS check below catches it for you.
FEATURE_VERSIONS = {
    "loudness": "loudness_v1",
    "brightness": "brightness_v1",
    "hits": "hits_v1",
}

# Every tunable number lives here (moves to config.yaml in Phase 6).
# The extractor passes these into the feature functions, so this dict is the
# single source of truth for what produced a curve.
PARAMS = {
    "hop_s": HOP_S,
    "loudness": {"window_s": 3.0, "floor": -70.0},
    "brightness": {"silence_lufs": -50.0, "n_fft": N_FFT, "stft_hop": STFT_HOP},
    "fast_loudness_window_s": 0.4,   # short window used for hits and usable range
    "usable_silence_lufs": -50.0,
    "hits": {"smooth_s": 1.0, "rise_s": 1.0, "min_rise_lu": 4.0,
             "min_gap_s": 4.0, "sustain_s": 2.0, "silence_lufs": -50.0},
}


def current_meta():
    """What an up-to-date output file must say about how it was made."""
    return json.loads(json.dumps({"versions": FEATURE_VERSIONS, "params": PARAMS}))


def features_path(track_id, directory=OUT_DIR):
    return Path(directory) / f"{int(track_id):04d}.npz"


def extract_track(y, sr):
    """Compute everything for one loaded track. y has shape (samples, channels)."""
    hop = PARAMS["hop_s"]
    n_expected = len(time_grid(len(y) / sr, hop))

    loudness = loudness_curve(y, sr, hop_s=hop, **PARAMS["loudness"])
    brightness = brightness_curve(
        y, sr, hop_s=hop, silence_lufs=PARAMS["brightness"]["silence_lufs"])
    fast = loudness_curve(y, sr, hop_s=hop, window_s=PARAMS["fast_loudness_window_s"])
    hits = detect_hits(fast, hop_s=hop, **PARAMS["hits"])
    start_s, end_s = usable_range(fast, hop_s=hop, silence_lufs=PARAMS["usable_silence_lufs"])

    for name, curve in (("loudness", loudness), ("brightness", brightness)):
        if len(curve) != n_expected:
            raise ValueError(f"{name} has {len(curve)} values, time grid has {n_expected}")
        if not np.all(np.isfinite(curve)):
            raise ValueError(f"{name} contains missing or infinite values")

    return {
        "loudness": loudness, "brightness": brightness, "hits": hits,
        "start_s": start_s, "end_s": end_s, "duration_s": len(y) / sr, "hop_s": hop,
    }


def save_features(path, data):
    """Write atomically, so an interrupted run never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hits = data["hits"]
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        loudness=data["loudness"], brightness=data["brightness"],
        hit_times=np.array([h[0] for h in hits], dtype=np.float64),
        hit_strengths=np.array([h[1] for h in hits], dtype=np.float64),
        start_s=data["start_s"], end_s=data["end_s"],
        duration_s=data["duration_s"], hop_s=data["hop_s"],
        meta=json.dumps(current_meta()),
    )
    tmp.replace(path)


def load_features(track_id, directory=OUT_DIR):
    """Read one track's saved features back (Phase 3 uses this to fill the database)."""
    with np.load(features_path(track_id, directory), allow_pickle=False) as f:
        return {
            "loudness": f["loudness"], "brightness": f["brightness"],
            "hits": list(zip(f["hit_times"].tolist(), f["hit_strengths"].tolist())),
            "start_s": float(f["start_s"]), "end_s": float(f["end_s"]),
            "duration_s": float(f["duration_s"]), "hop_s": float(f["hop_s"]),
            "meta": json.loads(str(f["meta"])),
        }


def status(path):
    """'missing', 'stale' (made with different versions/settings), or 'current'."""
    if not Path(path).exists():
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
    except Exception:
        return "stale"  # unreadable or corrupt: redo it
    return "current" if meta == current_meta() else "stale"


def run(manifest=MANIFEST, out_dir=OUT_DIR, failure_log=FAILURE_LOG, force=False, only=None):
    """Process the catalog. Returns {'done': n, 'skipped': n, 'failed': [(id, message)]}."""
    manifest = Path(manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found. Run this from the project root.")
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if only:
        rows = [r for r in rows if int(r["track_id"]) in set(only)]

    done, skipped, failures = 0, 0, []
    started = time.time()
    for n, row in enumerate(rows, 1):
        track_id = int(row["track_id"])
        path = features_path(track_id, out_dir)
        state = status(path)
        if state == "current" and not force:
            skipped += 1
            continue
        note = " (settings or version changed, recomputing)" if state == "stale" else ""
        t0 = time.time()
        try:
            y, sr = load_audio(row["file_path"])
            data = extract_track(y, sr)
            save_features(path, data)
            done += 1
            print(f"[{n}/{len(rows)}] track {track_id:04d} ok: {len(data['hits'])} hits, "
                  f"{time.time() - t0:.1f} s{note}", flush=True)
        except Exception as exc:  # keep going; one bad track must not stop the run
            failures.append((track_id, traceback.format_exc()))
            print(f"[{n}/{len(rows)}] track {track_id:04d} FAILED: {exc}", flush=True)

    Path(failure_log).parent.mkdir(parents=True, exist_ok=True)
    Path(failure_log).write_text(
        "".join(f"--- track {tid} ---\n{tb}\n" for tid, tb in failures) or "no failures\n",
        encoding="utf-8")
    print(f"\nDone: {done} processed, {skipped} already up to date, {len(failures)} failed "
          f"({time.time() - started:.0f} s). Output in {out_dir}/")
    if failures:
        print(f"Failed tracks: {[tid for tid, _ in failures]}. Details in {failure_log}")
    return {"done": done, "skipped": skipped, "failed": [(t, tb.splitlines()[-1]) for t, tb in failures]}


def main():
    parser = argparse.ArgumentParser(description="Extract loudness, brightness and hit points for every track.")
    parser.add_argument("--force", action="store_true", help="recompute every track")
    parser.add_argument("--only", type=int, nargs="+", metavar="ID", help="only these track ids")
    args = parser.parse_args()
    result = run(force=args.force, only=args.only)
    sys.exit(1 if result["failed"] else 0)


if __name__ == "__main__":
    main()