import numpy as np
import pytest
from scipy.signal import chirp

from shapesearch.audio import time_grid
from shapesearch.features.brightness import brightness_curve
from shapesearch.features.loudness import loudness_curve, usable_range

SR = 44100
rng = np.random.default_rng(0)


def stereo(x):
    return np.stack([x, x], axis=1).astype(np.float32)


def curves(y):
    return loudness_curve(y, SR), brightness_curve(y, SR)


def test_both_curves_match_time_grid():
    y = stereo(0.3 * rng.standard_normal(SR * 10))
    L, B = curves(y)
    assert len(L) == len(B) == len(time_grid(10.0)) == 100


def test_sine_sweep_brightness_rises_steadily():
    # exponential sweep 200 Hz -> 5000 Hz over 20 s
    t = np.arange(SR * 20) / SR
    y = stereo(0.5 * chirp(t, f0=200, t1=20, f1=5000, method="logarithmic"))
    L, B = curves(y)
    expected = np.log2(200) + (np.log2(5000) - np.log2(200)) * np.arange(len(B)) / len(B)
    assert B[-10] - B[10] > 4.0                      # rises ~4.6 octaves overall
    assert np.all(np.diff(B[5:-5]) > -0.05)          # never falls back
    assert np.max(np.abs(B[5:-5] - expected[5:-5])) < 0.3


def test_near_silence_does_not_produce_brightness_spikes():
    # 5 s of faint hiss (centroid ~11 kHz if taken at face value), then a 440 Hz tone
    hiss = 1e-5 * rng.standard_normal(SR * 5)
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR * 10) / SR)
    y = stereo(np.concatenate([hiss, tone]))
    L, B = curves(y)
    assert np.all(np.isfinite(B))
    assert np.max(np.abs(B - np.log2(440))) < 0.5    # intro filled from the tone


def test_fully_silent_track_is_rejected():
    y = np.zeros((SR * 5, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        brightness_curve(y, SR)


def test_usable_range_finds_silent_intro_and_outro():
    # 3 s silence, 10 s noise, 3 s silence
    x = np.concatenate([np.zeros(SR * 3), 0.2 * rng.standard_normal(SR * 10), np.zeros(SR * 3)])
    L = loudness_curve(stereo(x), SR, window_s=0.4)
    start_s, end_s = usable_range(L)
    assert 2.5 <= start_s <= 3.0   # music really starts at 3.0 s
    assert 13.0 <= end_s <= 13.5   # and ends at 13.0 s