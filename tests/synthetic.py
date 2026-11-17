"""Small made-up catalogs for the matching tests (shared by several test files)."""
import numpy as np

from shapesearch.matching.prep import CurveCache

N = 240                                   # 120 s at 2 values per second


def build_shape(seconds=30):
    """A slow build then a sustained peak, in loudness, with brightness rising too."""
    t = np.linspace(0, seconds, int(seconds * 2))
    return np.interp(t, [0, 0.67 * seconds, seconds], [-30, -12, -12]), np.interp(t, [0, seconds], [8.5, 10.5])


def fade_shape(seconds=30):
    t = np.linspace(0, seconds, int(seconds * 2))
    return np.interp(t, [0, seconds], [-12, -30]), np.interp(t, [0, seconds], [10.5, 8.5])


def background(rng):
    """Nearly flat noise: varies by less than the flatness guard."""
    return -20 + rng.normal(0, 0.3, N), 9 + rng.normal(0, 0.03, N)


def plant(rng, shape, at_s):
    loud, bright = background(rng)
    i = int(at_s * 2)
    loud[i:i + len(shape[0])] = shape[0]
    bright[i:i + len(shape[1])] = shape[1]
    return loud, bright, 0.0, N / 2


def make_cache(tracks, hits=None):
    """tracks: {id: (loudness, brightness, start_s, end_s)}; hits: {id: [(time_s, strength), ...]}"""
    curves, info = {}, {}
    for tid, (loud, bright, start, end) in tracks.items():
        curves[(tid, "loudness")] = np.asarray(loud, dtype=np.float32)
        curves[(tid, "brightness")] = np.asarray(bright, dtype=np.float32)
        info[tid] = {"start_s": start, "end_s": end, "duration_s": len(loud) / 2}
    hits = hits or {}
    return CurveCache(curves, info, {t: hits.get(t, []) for t in tracks}, ["loudness", "brightness"], {})


def two_peak_shape(first_at, second_at, seconds=30, width=2.0):
    """Quiet music with two loud peaks (triangles) at the given times; brightness follows loudness."""
    t = np.linspace(0, seconds, int(seconds * 2))
    bump = sum(np.clip(1 - np.abs(t - c) / width, 0, None) for c in (first_at, second_at))
    return -30 + 18 * bump, 8.5 + 2.0 * bump


def single_peak_shape(at, seconds=30, width=2.0):
    t = np.linspace(0, seconds, int(seconds * 2))
    bump = np.clip(1 - np.abs(t - at) / width, 0, None)
    return -30 + 18 * bump, 8.5 + 2.0 * bump