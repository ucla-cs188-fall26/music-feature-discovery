import numpy as np
import pyloudnorm as pyln

from shapesearch.audio import HOP_S, time_grid


def k_weight(y, sr):
    """Apply the K-weighting filter once to the whole track.
    y has shape (samples, channels). Uses pyloudnorm's filter stages."""
    meter = pyln.Meter(sr)
    out = y.astype(np.float64)
    for stage in meter._filters.values():
        for c in range(out.shape[1]):
            out[:, c] = stage.apply_filter(out[:, c])
    return out


def loudness_curve(y, sr, hop_s=HOP_S, window_s=3.0, floor=-70.0):
    """Short-term loudness (3 s window) every hop_s seconds, in LUFS.

    The window is centered on each grid time, so the curve lines up with
    what you hear at that moment. Near the start and end of the track the
    window is cut off and averaged over the audio that exists.
    Silence is clamped to `floor` so the log never returns -infinity.
    """
    n_samples, n_channels = y.shape
    if n_channels > 2:
        raise ValueError("loudness_curve supports mono or stereo only")

    times = time_grid(n_samples / sr, hop_s)
    kw = k_weight(y, sr)

    # Channel weight is 1.0 for left and right, so just add channel energies.
    power = np.sum(kw ** 2, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(power)])  # running total of energy

    lo = np.clip(np.round((times - window_s / 2) * sr).astype(int), 0, n_samples)
    hi = np.clip(np.round((times + window_s / 2) * sr).astype(int), 0, n_samples)
    mean_power = (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)
    mean_power = np.maximum(mean_power, 0.0)  # guard tiny negative rounding errors

    with np.errstate(divide="ignore"):
        lufs = -0.691 + 10.0 * np.log10(mean_power)
    return np.maximum(lufs, floor).astype(np.float32)

def usable_range(loudness, hop_s=HOP_S, silence_lufs=-50.0):
    """(start_s, end_s): where the track stops being silent at the start and
    end. The timeline is never cut; these are stored alongside it instead.

    Pass a loudness curve made with a short window, e.g.
    loudness_curve(y, sr, window_s=0.4). The default 3 s curve is smeared by
    about 1.5 s on each side, so it would report the music starting early."""
    loud = np.flatnonzero(loudness >= silence_lufs)
    if len(loud) == 0:
        raise ValueError("track has no usable (non-silent) audio")
    return float(loud[0] * hop_s), float((loud[-1] + 1) * hop_s)