import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from shapesearch.audio import HOP_S


def smooth(curve, window_s=1.5, hop_s=HOP_S, method="savgol", polyorder=2):
    """Smooth a curve over about window_s seconds.

    method="savgol": Savitzky-Golay filter. Keeps peaks and sharp rises
                     better than a plain average.
    method="moving": moving average.
    The returned curve has the same length as the input.
    """
    if method not in ("savgol", "moving"):
        raise ValueError(f"unknown smoothing method: {method!r}")
    x = np.asarray(curve, dtype=np.float64)
    n = int(round(window_s / hop_s))
    if n % 2 == 0:
        n += 1  # both filters want an odd window
    if n <= polyorder or n > len(x):
        return x.astype(np.float32)  # window too small or longer than the curve
    if method == "savgol":
        out = savgol_filter(x, n, polyorder, mode="nearest")
    else:
        out = uniform_filter1d(x, size=n, mode="nearest")
    return out.astype(np.float32)