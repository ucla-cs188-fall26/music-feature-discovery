from shapesearch.db import register_feature
from shapesearch.features.extract import FEATURE_VERSIONS, PARAMS


def feature_definitions():
    """The registry entries for every feature the extractor produces."""
    loud, bright, hop = PARAMS["loudness"], PARAMS["brightness"], PARAMS["hop_s"]
    return [
        dict(
            name="loudness",
            version=FEATURE_VERSIONS["loudness"],
            unit="LUFS",
            description=("How loud the music sounds at each moment, on the broadcast loudness "
                         "scale. Typical music sits between -20 and -8; silence is near -70."),
            method=(f"ITU-R BS.1770 short-term loudness: the audio is K-weighted (to match how "
                    f"ears weigh frequencies), its energy is averaged over a {loud['window_s']:g} s "
                    f"window centered on each point, then converted to a log scale. One value "
                    f"every {hop:g} s; silence is clamped to {loud['floor']:g}."),
            params={**loud, "hop_s": hop,
                    "hit_detection": {"loudness_window_s": PARAMS["fast_loudness_window_s"],
                                      **PARAMS["hits"]}},
        ),
        dict(
            name="brightness",
            version=FEATURE_VERSIONS["brightness"],
            unit="octaves",
            description=("How bright or sharp the sound is at each moment. Higher means more "
                         "high-frequency content, such as cymbals, filter sweeps and distortion."),
            method=(f"Spectral centroid (the loudness-weighted average frequency) of a "
                    f"{bright['n_fft']}-sample spectrogram, combined into {hop:g} s steps and "
                    f"converted to octaves (log base 2 of Hz). Near-silent moments (below "
                    f"{bright['silence_lufs']:g} LUFS) are filled in by interpolation."),
            params={**bright, "hop_s": hop},
        ),
    ]


def register_all(con):
    """Register (or update) every feature in the database."""
    for d in feature_definitions():
        register_feature(con, **d)