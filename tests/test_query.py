import json

import numpy as np
import pytest

from shapesearch.matching.query import (AnchorMatch, FeatureSketch, Query, QueryError, SearchResponse,
                                        SegmentResult, query_curves, sketch_to_curve)

PLAN_EXAMPLE = {
    "features": {
        "loudness": {"points": [[0, 0.1], [8, 0.8], [20, 0.9], [30, 0.6]], "weight": 0.5},
        "brightness": {"points": [[0, 0.2], [8, 0.9], [30, 0.7]], "weight": 0.5},
    },
    "duration_s": 30, "mode": "anchored", "anchors_s": [8], "top_k": 20,
}


def modified(**changes):
    data = json.loads(json.dumps(PLAN_EXAMPLE))
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def test_the_plans_example_query_is_accepted():
    q = Query.from_dict(PLAN_EXAMPLE)
    assert q.duration_s == 30 and q.mode == "anchored" and q.anchors_s == (8.0,) and q.top_k == 20
    assert q.features["loudness"].points[1] == (8.0, 0.8)
    assert q.active_features == ["loudness", "brightness"]


def test_defaults_when_optional_fields_are_left_out():
    q = Query.from_dict({"features": {"loudness": {"points": [[0, 0], [10, 1]]}}, "duration_s": 10})
    assert q.mode == "flexible" and q.top_k == 10 and q.features["loudness"].weight == 1.0


@pytest.mark.parametrize("data, message", [
    ("not a dict", "must be an object"),
    ({"duration_s": 30}, "missing required field 'features'"),
    ({"features": {"loudness": {"points": [[0, 0], [1, 1]]}}}, "missing required field 'duration_s'"),
    (modified(duraton_s=30), "unknown query field"),
    (modified(duration_s=1), "at least 2 seconds"),
    (modified(duration_s="30"), "duration_s must be a number"),
    (modified(mode="fuzzy"), "mode must be one of"),
    (modified(top_k=0), "top_k"),
    (modified(top_k=2.5), "top_k"),
    (modified(features={}), "at least one feature"),
    (modified(anchors_s=[40]), "outside the sketch"),
    (modified(anchors_s=[]), "needs at least one time"),
    (modified(mode="flexible"), "only applies in anchored mode"),
])
def test_bad_queries_get_clear_messages(data, message):
    with pytest.raises(QueryError, match=message):
        Query.from_dict(data)


def feature_with(**changes):
    body = {"points": [[0, 0.1], [8, 0.8], [30, 0.6]], "weight": 0.5}
    body.update(changes)
    return modified(features={"loudness": body}, mode="flexible", anchors_s=[])


@pytest.mark.parametrize("data, message", [
    (feature_with(points=[[0, 0.5]]), r"loudness: a sketch needs at least 2 points"),
    (feature_with(points=[[0, 0.1], [8, 1.5]]), r"loudness: point 2 height 1.5 must be between 0 and 1"),
    (feature_with(points=[[0, 0.1], [8, -0.2]]), r"point 2 height -0.2"),
    (feature_with(points=[[0, 0.1], [40, 0.5]]), r"point 2 time 40 s is outside the sketch"),
    (feature_with(points=[[0, 0.1], [10, 0.5], [5, 0.9]]), r"increasing time order; point 3"),
    (feature_with(points=[[0, 0.1], [10, 0.5], [10, 0.9]]), r"increasing time order"),
    (feature_with(points=[[0, 0.1], [10]]), r"point 2 must be \[seconds, height\]"),
    (feature_with(points=[[0, 0.1], ["5", 0.5]]), r"point 2 time must be a number"),
    (feature_with(points=[[0, 0.1], [5, float("nan")]]), r"point 2 height must be a finite number"),
    (feature_with(points="wiggly"), r"points must be a list"),
    (feature_with(weight=-1), r"weight must not be negative"),
    (feature_with(weight=0), r"weights are all zero"),
])
def test_bad_sketches_name_the_feature_and_the_point(data, message):
    with pytest.raises(QueryError, match=message):
        Query.from_dict(data)


def test_unknown_feature_lists_the_available_ones():
    with pytest.raises(QueryError, match=r"unknown feature 'loudnes'; available features: \['brightness', 'loudness'\]"):
        Query.from_dict(feature_with() | {"features": {"loudnes": {"points": [[0, 0], [1, 1]]}}},
                        known_features={"loudness", "brightness"})


def test_zero_weight_means_dont_care():
    data = json.loads(json.dumps(PLAN_EXAMPLE))
    data["features"]["brightness"]["weight"] = 0
    q = Query.from_dict(data)
    assert q.active_features == ["loudness"]
    assert list(query_curves(q)) == ["loudness"]


def test_four_point_sketch_becomes_60_values_for_a_30_second_query():
    q = Query.from_dict(PLAN_EXAMPLE)
    curve = sketch_to_curve(q.features["loudness"], q.duration_s)
    assert curve.shape == (60,)
    assert curve.min() >= 0.1 - 1e-9 and curve.max() <= 0.9 + 1e-9


def test_sketch_curve_follows_the_drawn_lines():
    sketch = FeatureSketch([(0, 0.0), (10, 1.0)])
    curve = sketch_to_curve(sketch, 10)
    assert len(curve) == 20
    assert np.all(np.diff(curve) > 0)                      # a straight rise
    assert curve[0] == pytest.approx(0.025) and curve[-1] == pytest.approx(0.975)   # read at cell centres
    assert np.allclose(np.diff(curve), 0.05)               # even steps


def test_curve_holds_the_edge_heights_outside_the_drawn_points():
    sketch = FeatureSketch([(2, 0.2), (4, 0.8)])
    curve = sketch_to_curve(sketch, 6)
    assert np.all(curve[:4] == 0.2) and np.all(curve[-4:] == 0.8)


def test_results_convert_to_plain_json():
    result = SegmentResult(track_id=3, start_s=72.0, end_s=102.0, overall_score=87.5,
                           feature_scores={"loudness": 90.0, "brightness": 85.0},
                           anchors=[AnchorMatch(8.0, 80.3, 12.5)], path=[(0, 0), (1, 1)])
    response = SearchResponse([result], {"loudness": 0.5, "brightness": 0.5}, "anchored", 0.42)
    decoded = json.loads(json.dumps(response.to_dict()))
    assert decoded["results"][0]["feature_scores"]["loudness"] == 90.0
    assert decoded["results"][0]["anchors"][0]["hit_time_s"] == 80.3
    assert decoded["weights"] == {"loudness": 0.5, "brightness": 0.5}