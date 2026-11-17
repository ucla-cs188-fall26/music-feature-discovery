import numpy as np
import pytest

from shapesearch.features.smoothing import smooth

rng = np.random.default_rng(0)


@pytest.mark.parametrize("method", ["savgol", "moving"])
def test_keeps_length_and_reduces_jitter(method):
    clean = np.linspace(-30, -10, 600)
    noisy = clean + rng.normal(0, 2, 600)
    out = smooth(noisy, window_s=1.5, method=method)
    assert len(out) == len(noisy)
    assert np.std(out - clean) < 0.5 * np.std(noisy - clean)


def test_savgol_keeps_peaks_sharper_than_moving_average():
    t = np.arange(300) * 0.1
    peak = np.exp(-((t - 15) ** 2) / (2 * 0.4 ** 2))
    assert smooth(peak, 1.5, method="savgol").max() > smooth(peak, 1.5, method="moving").max()


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        smooth(np.zeros(100), method="magic")