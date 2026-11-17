import librosa
import numpy as np

from shapesearch.audio import HOP_S, time_grid
from shapesearch.features.loudness import loudness_curve

N_FFT = 2048
STFT_HOP = 512  # samples between spectrogram frames


def centroid_per_bin(mono, sr, n_grid, hop_s=HOP_S, n_fft=N_FFT, hop_length=STFT_HOP):
    """Spectral centroid in Hz, one value per grid time.

    Per spectrogram frame the centroid is  sum(f * |X|) / sum(|X|).  Frames are
    grouped into 0.1 s bins (bin i is centered on i * hop_s) and combined by
    adding the numerators and denominators separately. That is the same as
    averaging the frame centroids weighted by how loud each frame is, so a
    quiet frame cannot drag a bin's brightness around.
    Bins with no energy at all come back as NaN.
    """
    mag = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    weighted = (freqs[:, None] * mag).sum(axis=0)  # sum of f * |X|, per frame
    total = mag.sum(axis=0)                        # sum of |X|, per frame

    frame_times = np.arange(mag.shape[1]) * hop_length / sr
    idx = np.floor((frame_times + hop_s / 2) / hop_s).astype(int)
    keep = idx < n_grid
    num = np.bincount(idx[keep], weights=weighted[keep], minlength=n_grid)
    den = np.bincount(idx[keep], weights=total[keep], minlength=n_grid)

    out = np.full(n_grid, np.nan)
    np.divide(num, den, out=out, where=den > 1e-10)
    return out


def fill_missing(values):
    """Fill NaNs by linear interpolation. A silent intro or outro has no
    neighbor on one side, so it holds the nearest valid value."""
    bad = np.isnan(values)
    if bad.all():
        raise ValueError("track has no usable (non-silent) audio")
    x = np.arange(len(values))
    return np.interp(x, x[~bad], values[~bad])


def brightness_curve(y, sr, hop_s=HOP_S, silence_lufs=-50.0):
    """Brightness in octaves (log2 of the spectral centroid in Hz), one value
    per grid time.

    Near-silent cells (loudness below `silence_lufs`) have a meaningless
    centroid, so they are marked missing and filled by interpolation. Silence
    is judged with loudness measured over just that 0.1 s cell. The 3 s
    short-term curve would smear about 1.5 s into silence on each side.
    """
    n_grid = len(time_grid(y.shape[0] / sr, hop_s))
    mono = y.mean(axis=1)  # brightness is measured on a mono mix
    centroid_hz = centroid_per_bin(mono, sr, n_grid, hop_s)
    octaves = np.log2(np.maximum(centroid_hz, 1.0))  # NaN stays NaN

    cell_loudness = loudness_curve(y, sr, hop_s=hop_s, window_s=hop_s)
    octaves[cell_loudness < silence_lufs] = np.nan
    return fill_missing(octaves).astype(np.float32)