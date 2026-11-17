import math
from dataclasses import asdict, dataclass, field

import numpy as np

SKETCH_RATE = 2          # curve values per second, for sketches and thinned stored curves
MIN_DURATION_S = 2.0
MODES = ("flexible", "anchored")
_QUERY_KEYS = {"features", "duration_s", "mode", "anchors_s", "top_k"}


class QueryError(ValueError):
    """A search query is malformed. The message says what to fix."""


def _num(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueryError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise QueryError(f"{what} must be a finite number, got {value!r}")
    return float(value)


# ---------------------------------------------------------------- the query

@dataclass(frozen=True)
class FeatureSketch:
    """One feature's drawn shape: points are (seconds, height 0 to 1)."""
    points: tuple
    weight: float = 1.0   # 0 means "don't care": the feature is ignored

    def __post_init__(self):
        if not isinstance(self.points, (list, tuple)):
            raise QueryError("points must be a list of [seconds, height] pairs")
        parsed = []
        for i, p in enumerate(self.points, 1):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise QueryError(f"point {i} must be [seconds, height], got {p!r}")
            parsed.append((_num(p[0], f"point {i} time"), _num(p[1], f"point {i} height")))
        object.__setattr__(self, "points", tuple(parsed))
        object.__setattr__(self, "weight", _num(self.weight, "weight"))


@dataclass(frozen=True)
class Query:
    features: dict                 # {feature name: FeatureSketch}
    duration_s: float
    mode: str = "flexible"
    anchors_s: tuple = ()
    top_k: int = 10

    def __post_init__(self):
        duration = _num(self.duration_s, "duration_s")
        if duration < MIN_DURATION_S:
            raise QueryError(f"duration_s must be at least {MIN_DURATION_S:g} seconds, got {duration:g}")
        object.__setattr__(self, "duration_s", duration)

        if self.mode not in MODES:
            raise QueryError(f"mode must be one of {list(MODES)}, got {self.mode!r}")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise QueryError(f"top_k must be a whole number of at least 1, got {self.top_k!r}")

        if not isinstance(self.features, dict) or not self.features:
            raise QueryError("features must contain at least one feature sketch")
        for name, sketch in self.features.items():
            self._check_sketch(name, sketch, duration)
        if all(s.weight == 0 for s in self.features.values()):
            raise QueryError("weights are all zero; at least one feature needs a weight above 0")

        anchors = tuple(sorted(_num(a, f"anchor {i}") for i, a in enumerate(self.anchors_s, 1)))
        for a in anchors:
            if not 0 <= a <= duration:
                raise QueryError(f"anchor at {a:g} s is outside the sketch (0 to {duration:g} s)")
        if self.mode == "anchored" and not anchors:
            raise QueryError("anchored mode needs at least one time in anchors_s")
        if self.mode == "flexible" and anchors:
            raise QueryError("anchors_s only applies in anchored mode; set mode to 'anchored' or remove them")
        object.__setattr__(self, "anchors_s", anchors)

    @staticmethod
    def _check_sketch(name, sketch, duration):
        if not isinstance(sketch, FeatureSketch):
            raise QueryError(f"{name}: expected a FeatureSketch, got {type(sketch).__name__}")
        pts = sketch.points
        if len(pts) < 2:
            raise QueryError(f"{name}: a sketch needs at least 2 points, got {len(pts)}")
        for i, (t, h) in enumerate(pts, 1):
            if not 0 <= t <= duration:
                raise QueryError(f"{name}: point {i} time {t:g} s is outside the sketch (0 to {duration:g} s)")
            if not 0 <= h <= 1:
                raise QueryError(f"{name}: point {i} height {h:g} must be between 0 and 1")
            if i > 1 and t <= pts[i - 2][0]:
                raise QueryError(f"{name}: points must be in increasing time order; point {i} "
                                 f"(t={t:g}) is not after point {i - 1} (t={pts[i - 2][0]:g})")
        if sketch.weight < 0:
            raise QueryError(f"{name}: weight must not be negative, got {sketch.weight:g}")

    @property
    def active_features(self):
        """Names of the features that count (weight above 0), in query order."""
        return [n for n, s in self.features.items() if s.weight > 0]

    @classmethod
    def from_dict(cls, data, known_features=None):
        """Build a Query from the JSON shape. Pass known_features (e.g. the names
        in the database) to reject unknown feature names with a helpful message."""
        if not isinstance(data, dict):
            raise QueryError("a query must be an object with 'features' and 'duration_s'")
        extra = set(data) - _QUERY_KEYS
        if extra:
            raise QueryError(f"unknown query field(s) {sorted(extra)}; allowed: {sorted(_QUERY_KEYS)}")
        for required in ("features", "duration_s"):
            if required not in data:
                raise QueryError(f"missing required field '{required}'")
        raw = data["features"]
        if not isinstance(raw, dict):
            raise QueryError("features must be an object like {'loudness': {'points': [...]}}")

        sketches = {}
        for name, body in raw.items():
            if known_features is not None and name not in known_features:
                raise QueryError(f"unknown feature {name!r}; available features: {sorted(known_features)}")
            if not isinstance(body, dict) or "points" not in body:
                raise QueryError(f"{name}: expected an object with 'points' (and optionally 'weight')")
            try:
                sketches[name] = FeatureSketch(body["points"], body.get("weight", 1.0))
            except QueryError as exc:
                raise QueryError(f"{name}: {exc}") from None
        return cls(features=sketches, duration_s=data["duration_s"], mode=data.get("mode", "flexible"),
                   anchors_s=tuple(data.get("anchors_s", ())), top_k=data.get("top_k", 10))



@dataclass
class AnchorMatch:
    """Which stored hit point satisfied an anchor."""
    anchor_s: float        # the anchor's time in the sketch
    hit_time_s: float      # the hit's time in the track's own timeline
    hit_strength: float    # size of the hit's rise, in LU


@dataclass
class SegmentResult:
    track_id: int
    start_s: float                                   # times in the original audio file
    end_s: float
    overall_score: float                             # 0 to 100
    feature_scores: dict                             # {feature name: 0 to 100}
    anchors: list = field(default_factory=list)      # [AnchorMatch]
    path: list = field(default_factory=list)         # [(sketch index, window index)], 2 values per second

    def to_dict(self):
        return asdict(self)


@dataclass
class SearchResponse:
    results: list                                    # [SegmentResult], best first
    weights: dict                                    # the weights actually used, always returned
    mode: str
    search_time_s: float = 0.0

    def to_dict(self):
        return asdict(self)


def sketch_to_curve(sketch, duration_s, rate=SKETCH_RATE):
    """Turn a drawn sketch into a curve: connect the points with straight lines
    and sample at `rate` values per second. Value i represents the half second
    [i/2, (i+1)/2), so it is read at that cell's centre. Before the first point
    and after the last, the curve holds the nearest point's height."""
    n = int(round(duration_s * rate))
    t = (np.arange(n) + 0.5) / rate
    times, heights = zip(*sketch.points)
    return np.interp(t, times, heights)


def query_curves(query):
    """{feature name: sketch curve} for every feature that counts (weight above 0)."""
    return {name: sketch_to_curve(query.features[name], query.duration_s)
            for name in query.active_features}