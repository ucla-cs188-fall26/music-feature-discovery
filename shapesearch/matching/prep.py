import numpy as np

from shapesearch.db import list_features, list_tracks, load_curve, load_hits
from shapesearch.features.smoothing import smooth

THIN_FACTOR = 5          # 10 values per second -> 2 values per second
SMOOTH_S = 1.5           # smoothing window applied to stored curves at startup

# Flatness guard (r_min): a window that varies less than this counts as "nearly
# flat", so it is not stretched up to the full 0-1 range. In the feature's units.
# A new feature should add its own entry here; others fall back to DEFAULT_MIN_RANGE.
MIN_RANGE = {"loudness": 3.0, "brightness": 0.3}
DEFAULT_MIN_RANGE = 1.0
SKETCH_MIN_RANGE = 0.2   # same idea for a sketch, whose heights are on a 0-1 scale


def thin(curve, factor=THIN_FACTOR):
    """Average groups of `factor` values. A leftover partial group at the end is dropped."""
    curve = np.asarray(curve, dtype=np.float64)
    n = len(curve) // factor
    return curve[: n * factor].reshape(n, factor).mean(axis=1).astype(np.float32)


class CurveCache:
    """Every track's smoothed, thinned curves, kept in memory.

    curve(track_id, feature) -> array with 2 values per second
    track_info(track_id)     -> {'start_s', 'end_s', 'duration_s'} (the usable range)
    hits(track_id)           -> [(time_s, strength), ...]
    """

    def __init__(self, curves, tracks, hits, feature_names, settings):
        self._curves = curves
        self._tracks = tracks
        self._hits = hits
        self.feature_names = feature_names
        self.settings = settings

    @classmethod
    def from_database(cls, con, smooth_s=SMOOTH_S, smooth_method="savgol"):
        features = [f["name"] for f in list_features(con)]
        curves, tracks, hits = {}, {}, {}
        for t in list_tracks(con):
            tid = t["track_id"]
            tracks[tid] = {"start_s": t["start_s"], "end_s": t["end_s"], "duration_s": t["duration_s"]}
            lengths = set()
            for name in features:
                raw = load_curve(con, tid, name)
                curves[(tid, name)] = thin(smooth(raw, smooth_s, method=smooth_method))
                lengths.add(len(curves[(tid, name)]))
            if len(lengths) > 1:
                raise ValueError(f"track {tid}: features have different lengths {sorted(lengths)}; "
                                 f"run scripts/check_integrity.py")
            hits[tid] = load_hits(con, tid)
        settings = {"smooth_s": smooth_s, "smooth_method": smooth_method, "thin_factor": THIN_FACTOR}
        return cls(curves, tracks, hits, features, settings)

    @property
    def track_ids(self):
        return sorted(self._tracks)

    def curve(self, track_id, feature):
        return self._curves[(track_id, feature)]

    def track_info(self, track_id):
        return self._tracks[track_id]

    def hits(self, track_id):
        return self._hits[track_id]

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self._curves.values())

    def __len__(self):
        return len(self._tracks)


def min_range_for(feature):
    return MIN_RANGE.get(feature, DEFAULT_MIN_RANGE)


def normalize(x, min_range):
    """Rescale to 0-1 along the last axis, so shapes are compared and not absolute levels:

        x' = (x - min x) / max(max x - min x, min_range)

    The guard keeps a nearly flat curve from being stretched into exaggerated
    noise: if it varies by less than min_range, it stays proportionally small.
    Works on one curve or on a whole stack of windows (one per row)."""
    x = np.asarray(x)
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, min_range)


def normalize_feature(x, feature):
    """normalize() with the flatness guard for this feature's units."""
    return normalize(x, min_range_for(feature))


def normalize_sketch(curve, min_range=SKETCH_MIN_RANGE):
    """normalize() for a sketch curve, whose heights are on a 0-1 scale."""
    return normalize(curve, min_range)