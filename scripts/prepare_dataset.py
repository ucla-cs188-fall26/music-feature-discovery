"""Standardize raw audio and build data/manifest.csv."""
import csv
import subprocess
from pathlib import Path

import soundfile as sf

RAW, OUT = Path("data/raw"), Path("data/processed")
DRAFT, MANIFEST = Path("data/id_map.csv"), Path("data/manifest.csv")
COLUMNS = ["track_id", "title", "artist", "source", "source_url",
           "license", "file_path", "duration_s", "genre"]

OUT.mkdir(parents=True, exist_ok=True)
rows = []

# we update the id_map with new 
with open(DRAFT, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        row["title"] = Path(row["original_filename"]).stem   # name = original filename
        stem = f"{int(row['track_id']):04d}"
        raw = next(iter(sorted(RAW.glob(stem + ".*"))), None)
        if raw is None:
            print(f"MISSING raw file for track {stem}")
            continue
        out = OUT / f"{stem}.wav"
        if not out.exists():  # skip tracks already converted
            # the subprocess converts each of the .mp3 files to .WAV files. It preserves stero audio.
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                 "-ar", "44100", "-ac", "2", str(out)],
                check=True,
            )
        info = sf.info(out)
        row["file_path"] = str(out)
        row["duration_s"] = round(info.frames / info.samplerate, 2)
        if not 60 <= row["duration_s"] <= 480:
            print(f"WARNING track {stem}: {row['duration_s']} s is outside 1-8 min")
        rows.append(row)

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} tracks to {MANIFEST}")