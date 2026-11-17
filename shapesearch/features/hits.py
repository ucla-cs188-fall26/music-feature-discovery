import numpy as np
from scipy.signal import find_peaks

from shapesearch.audio import HOP_S
from shapesearch.features.smoothing import smooth


def detect_hits(loudness, hop_s=HOP_S, smooth_s=1.0, rise_s=1.0,
                min_rise_lu=4.0, min_gap_s=4.0, sustain_s=2.0, silence_lufs=-50.0):
    Ls = smooth(loudness, smooth_s, hop_s)
    k = int(round(rise_s / hop_s))
    d = np.zeros_like(Ls)
    d[k:] = Ls[k:] - Ls[:-k]

    peaks, _ = find_peaks(d, height=min_rise_lu)
    hold = int(round(sustain_s / hop_s))

    candidates = []
    for i in peaks:
        if Ls[i - k] < silence_lufs:
            continue  # rise out of silence: the track simply started
        floor_after = Ls[i - k] + 0.5 * d[i]
        if Ls[i: i + hold + 1].min() < floor_after:
            continue  # did not stay up
        candidates.append(((i - k / 2) * hop_s, float(d[i])))

    accepted = []
    for t, strength in sorted(candidates, key=lambda c: -c[1]):  # strongest first
        if all(abs(t - a) >= min_gap_s for a, _ in accepted):
            accepted.append((float(t), strength))
    return sorted(accepted)