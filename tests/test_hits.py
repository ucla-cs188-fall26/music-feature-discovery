import numpy as np

from shapesearch.features.hits import detect_hits
from shapesearch.features.loudness import loudness_curve

SR = 44100
rng = np.random.default_rng(0)


def hits_for(amplitude):
    """amplitude: per-sample volume envelope. Returns detected hits."""
    x = amplitude * rng.standard_normal(len(amplitude))
    y = np.stack([x, x], axis=1).astype(np.float32)
    return detect_hits(loudness_curve(y, SR, window_s=0.4))


def envelope(seconds, steps):
    """steps: [(time_s, amplitude), ...] - volume jumps to `amplitude` at `time_s`."""
    t = np.arange(int(seconds * SR)) / SR
    amp = np.full(len(t), steps[0][1])
    for time_s, a in steps[1:]:
        amp[t >= time_s] = a
    return amp


def test_sharp_drop_is_found_at_the_right_time():
    hits = hits_for(envelope(20, [(0, 0.01), (10.0, 0.2)]))   # +26 dB at 10 s
    assert len(hits) == 1
    time_s, strength = hits[0]
    assert abs(time_s - 10.0) <= 0.3
    assert strength > 10


def test_gradual_fade_is_not_a_hit():
    t = np.arange(SR * 20) / SR
    amp = 0.02 * 10 ** (np.clip((t - 5) / 10, 0, 1) * 12 / 20)   # +12 dB over 10 s
    assert hits_for(amp) == []


def test_getting_quieter_is_not_a_hit():
    assert hits_for(envelope(20, [(0, 0.2), (10.0, 0.01)])) == []


def test_flat_noise_has_no_hits():
    assert hits_for(envelope(20, [(0, 0.1)])) == []


def test_short_burst_that_dies_away_is_not_a_hit():
    # half a second of loud noise in quiet noise: rises sharply but does not stay up
    amp = envelope(20, [(0, 0.01), (10.0, 0.3), (10.5, 0.01)])
    assert hits_for(amp) == []


def test_drops_closer_than_4_seconds_merge_into_one_hit():
    hits = hits_for(envelope(25, [(0, 0.01), (10.0, 0.05), (12.0, 0.25)]))
    assert len(hits) == 1


def test_drops_far_apart_are_both_found():
    hits = hits_for(envelope(30, [(0, 0.01), (10.0, 0.05), (18.0, 0.25)]))
    assert len(hits) == 2
    assert abs(hits[0][0] - 10.0) <= 0.3 and abs(hits[1][0] - 18.0) <= 0.3


def test_music_starting_from_silence_is_not_a_hit():
    amp = envelope(20, [(0, 0.0), (5.0, 0.2)])
    assert hits_for(amp) == []