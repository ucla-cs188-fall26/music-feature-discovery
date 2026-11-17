import numpy as np
import pyloudnorm as pyln

from shapesearch.audio import time_grid
from shapesearch.features.loudness import loudness_curve

SR = 44100
rng = np.random.default_rng(0)


def noise(seconds, amp=0.1):
    return (amp * rng.standard_normal((int(seconds * SR), 2))).astype(np.float32)


def test_curve_length_matches_time_grid():
    y = noise(10.0)
    assert len(loudness_curve(y, SR)) == len(time_grid(10.0)) == 100


def test_silence_is_clamped_to_floor():
    y = np.zeros((SR * 5, 2), dtype=np.float32)
    assert np.all(loudness_curve(y, SR) == -70.0)


def test_12db_fade_rises_about_12_lu():
    # 5 s steady, 10 s ramp up by 12 dB (linear in dB), 5 s steady
    n = SR * 20
    t = np.arange(n) / SR
    gain_db = np.clip((t - 5) / 10, 0, 1) * 12
    amp = 0.02 * 10 ** (gain_db / 20)
    y = (amp[:, None] * rng.standard_normal((n, 2))).astype(np.float32)
    L = loudness_curve(y, SR)
    rise = L[170] - L[30] 
    assert abs(rise - 12) < 0.5


def test_matches_pyloudnorm_on_steady_noise():
    y = noise(10.0)
    L = loudness_curve(y, SR)
    segment = y[int(3.5 * SR): int(6.5 * SR)] 
    reference = pyln.Meter(SR).integrated_loudness(segment)
    assert abs(L[50] - reference) < 0.3