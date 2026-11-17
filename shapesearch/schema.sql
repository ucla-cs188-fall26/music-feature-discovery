-- One row per track. start_s / end_s mark where the usable audio begins and ends.
CREATE TABLE IF NOT EXISTS tracks (
    track_id    INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    artist      TEXT,
    genre       TEXT,
    source      TEXT,
    source_url  TEXT,
    license     TEXT,                 -- optional in this prototype
    file_path   TEXT NOT NULL,
    duration_s  REAL NOT NULL,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL
);

-- The feature registry: every available curve, with its unit and exact method.
-- A new feature (e.g. drum-stem loudness) is one new row here, not a new column.
CREATE TABLE IF NOT EXISTS features (
    feature_id   INTEGER PRIMARY KEY,
    name         TEXT UNIQUE NOT NULL,   -- 'loudness', 'brightness'
    version      TEXT NOT NULL,          -- 'loudness_v1'
    unit         TEXT NOT NULL,          -- 'LUFS', 'octaves'
    description  TEXT NOT NULL,          -- plain words for users
    method       TEXT NOT NULL,          -- e.g. 'ITU-R BS.1770 short-term, 3 s window'
    params_json  TEXT NOT NULL           -- exact settings, e.g. window and hop sizes
);

-- One curve per track per feature, stored as a packed array of float32 numbers.
CREATE TABLE IF NOT EXISTS curves (
    track_id     INTEGER NOT NULL REFERENCES tracks(track_id),
    feature_id   INTEGER NOT NULL REFERENCES features(feature_id),
    hop_s        REAL NOT NULL,          -- 0.1
    n_values     INTEGER NOT NULL,
    values_blob  BLOB NOT NULL,          -- float32 array, raw (unsmoothed)
    PRIMARY KEY (track_id, feature_id)
);

-- Detected hit points (sudden rises that stay up).
CREATE TABLE IF NOT EXISTS hit_points (
    hit_id      INTEGER PRIMARY KEY,
    track_id    INTEGER NOT NULL REFERENCES tracks(track_id),
    time_s      REAL NOT NULL,
    strength    REAL NOT NULL,           -- size of the rise, in LU
    feature_id  INTEGER NOT NULL REFERENCES features(feature_id)   -- which curve it came from
);

CREATE INDEX IF NOT EXISTS idx_hits_track  ON hit_points(track_id);
CREATE INDEX IF NOT EXISTS idx_tracks_genre ON tracks(genre);