"""All database access. The rest of the code never writes SQL directly."""
import json
import sqlite3
from pathlib import Path

import numpy as np

from shapesearch.audio import HOP_S


DB_PATH = Path("data/shapes.db")
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(path=DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")   # SQLite ignores links between tables unless asked
    con.row_factory = sqlite3.Row             # rows readable by column name
    return con


def create_schema(con):
    con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    con.commit()


def table_counts(con):
    names = [r["name"] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {n: con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


# ---------------------------------------------------------------- tracks

def add_track(con, track_id, title, file_path, duration_s, start_s, end_s,
              artist=None, genre=None, source=None, source_url=None, license=None):
    """Add a track, or update it if this track_id already exists.
    track_id comes from the manifest, so 0007.wav is always track 7."""
    con.execute(
        """INSERT INTO tracks (track_id, title, artist, genre, source, source_url,
                               license, file_path, duration_s, start_s, end_s)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(track_id) DO UPDATE SET
               title = excluded.title, artist = excluded.artist, genre = excluded.genre,
               source = excluded.source, source_url = excluded.source_url,
               license = excluded.license, file_path = excluded.file_path,
               duration_s = excluded.duration_s, start_s = excluded.start_s,
               end_s = excluded.end_s""",
        (int(track_id), title, artist, genre, source, source_url, license,
         file_path, float(duration_s), float(start_s), float(end_s)))
    con.commit()
    return int(track_id)


def get_track(con, track_id):
    """One track as a dict, or None if there is no such track."""
    row = con.execute("SELECT * FROM tracks WHERE track_id = ?", (int(track_id),)).fetchone()
    return dict(row) if row else None


def list_tracks(con, genre=None):
    """All tracks (optionally only one genre), ordered by track_id."""
    if genre is None:
        rows = con.execute("SELECT * FROM tracks ORDER BY track_id")
    else:
        rows = con.execute("SELECT * FROM tracks WHERE genre = ? ORDER BY track_id", (genre,))
    return [dict(r) for r in rows]


# -------------------------------------------------------------- features

def register_feature(con, name, version, unit, description, method, params=None):
    """Add a feature to the registry, or update it if the name already exists.
    Returns its feature_id (which never changes for a given name)."""
    con.execute(
        """INSERT INTO features (name, version, unit, description, method, params_json)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
               version = excluded.version, unit = excluded.unit,
               description = excluded.description, method = excluded.method,
               params_json = excluded.params_json""",
        (name, version, unit, description, method, json.dumps(params or {}, sort_keys=True)))
    con.commit()
    return _feature_id(con, name)


def list_features(con):
    """Every registered feature, with its settings as a dict under 'params'."""
    out = []
    for row in con.execute("SELECT * FROM features ORDER BY feature_id"):
        d = dict(row)
        d["params"] = json.loads(d.pop("params_json"))
        out.append(d)
    return out


def _feature_id(con, name):
    row = con.execute("SELECT feature_id FROM features WHERE name = ?", (name,)).fetchone()
    if row is None:
        known = [r["name"] for r in con.execute("SELECT name FROM features ORDER BY feature_id")]
        raise KeyError(f"unknown feature {name!r}; registered features: {known}")
    return row["feature_id"]


def _require_track(con, track_id):
    if get_track(con, track_id) is None:
        raise KeyError(f"unknown track {track_id}")


# ---------------------------------------------------------------- curves

def save_curve(con, track_id, feature_name, values, hop_s=HOP_S):
    """Store one curve (raw, unsmoothed) as a packed float32 array.
    Replaces any curve already saved for this track and feature."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("a curve must be a non-empty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("a curve must not contain missing or infinite values")
    _require_track(con, track_id)
    feature_id = _feature_id(con, feature_name)

    blob = np.ascontiguousarray(values, dtype="<f4").tobytes()   # little-endian float32
    con.execute(
        """INSERT INTO curves (track_id, feature_id, hop_s, n_values, values_blob)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(track_id, feature_id) DO UPDATE SET
               hop_s = excluded.hop_s, n_values = excluded.n_values,
               values_blob = excluded.values_blob""",
        (int(track_id), feature_id, float(hop_s), len(values), blob))
    con.commit()


def load_curve(con, track_id, feature_name):
    """One curve as a float32 NumPy array (a writable copy)."""
    row = con.execute(
        """SELECT n_values, values_blob FROM curves
           WHERE track_id = ? AND feature_id = ?""",
        (int(track_id), _feature_id(con, feature_name))).fetchone()
    if row is None:
        raise KeyError(f"no {feature_name!r} curve saved for track {track_id}")
    values = np.frombuffer(row["values_blob"], dtype="<f4").astype(np.float32)
    if len(values) != row["n_values"]:
        raise ValueError(f"stored curve for track {track_id} is damaged "
                         f"({len(values)} values, expected {row['n_values']})")
    return values


# ------------------------------------------------------------------ hits

def save_hits(con, track_id, feature_name, hits):
    """Store a track's hit points: a list of (time_s, strength).
    Replaces any hits already saved for this track and feature."""
    _require_track(con, track_id)
    feature_id = _feature_id(con, feature_name)
    con.execute("DELETE FROM hit_points WHERE track_id = ? AND feature_id = ?",
                (int(track_id), feature_id))
    con.executemany(
        "INSERT INTO hit_points (track_id, time_s, strength, feature_id) VALUES (?, ?, ?, ?)",
        [(int(track_id), float(t), float(s), feature_id) for t, s in hits])
    con.commit()


def load_hits(con, track_id, feature_name="loudness"):
    """A track's hit points as [(time_s, strength), ...], earliest first."""
    rows = con.execute(
        """SELECT time_s, strength FROM hit_points
           WHERE track_id = ? AND feature_id = ? ORDER BY time_s""",
        (int(track_id), _feature_id(con, feature_name)))
    return [(r["time_s"], r["strength"]) for r in rows]