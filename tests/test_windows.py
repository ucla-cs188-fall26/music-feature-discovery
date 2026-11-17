import pytest

from shapesearch.matching.windows import generate_windows


def test_plan_example_counts_for_a_100_second_track():
    windows = generate_windows(n_cells=200, usable_start_s=0, usable_end_s=100, target_cells=60)
    by_scale = {s: [w for w in windows if w.scale == s] for s in (0.8, 1.0, 1.25)}
    assert [len(by_scale[s]) for s in (0.8, 1.0, 1.25)] == [77, 71, 63]   # starts 0..76, 0..70, 0..62
    assert {w.n for w in by_scale[0.8]} == {48} and {w.n for w in by_scale[1.25]} == {75}
    assert by_scale[1.25][0].end_s == 37.5                                 # 1.25 x 30 s


def test_windows_start_on_whole_seconds_and_slices_match_times():
    windows = generate_windows(200, 0, 100, 60)
    for w in windows:
        assert w.start_s == int(w.start_s)
        assert w.start_idx == round(w.start_s * 2) and w.slice == slice(w.start_idx, w.start_idx + w.n)
        assert w.end_s == pytest.approx(w.start_s + w.n / 2)


def test_silent_edges_are_never_searched():
    windows = generate_windows(n_cells=200, usable_start_s=10.4, usable_end_s=90, target_cells=60)
    assert min(w.start_s for w in windows) == 11.0                         # first whole second at or after 10.4
    assert max(w.end_s for w in windows) <= 90.0
    assert all(w.start_s >= 10.4 for w in windows)


def test_windows_never_run_past_the_end_of_the_curve():
    windows = generate_windows(n_cells=100, usable_start_s=0, usable_end_s=60, target_cells=60)   # curve is only 50 s long
    assert windows and all(w.start_idx + w.n <= 100 for w in windows)


def test_track_shorter_than_the_sketch_gives_no_windows():
    assert generate_windows(n_cells=40, usable_start_s=0, usable_end_s=20, target_cells=60) == []


def test_track_that_fits_only_the_shortest_length():
    windows = generate_windows(n_cells=100, usable_start_s=0, usable_end_s=26, target_cells=60)   # 26 s usable
    assert {w.scale for w in windows} == {0.8}                                                     # only 24 s fits


def test_duplicate_lengths_are_not_repeated():
    windows = generate_windows(n_cells=100, usable_start_s=0, usable_end_s=50, target_cells=2)
    assert len({(w.start_s, w.n) for w in windows}) == len(windows)