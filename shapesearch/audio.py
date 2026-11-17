import numpy as np
import soundfile as sf

HOP_S = 0.1  # one curve value every 0.1 s


def load_audio(path):
    # Processed WAV returns (y, sr) with y shaped (samples, channels). 
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return y, sr


def time_grid(duration_s, hop_s=HOP_S):
    n = int(duration_s / hop_s + 1e-6)  # tiny epsilon guards float rounding
    return np.arange(n) * hop_s