import numpy as np

from shapesearch.audio import HOP_S, time_grid


def find_problems(con):
    """Return a list of plain-English problems. An empty list means all is well."""
    problems = []

    # 1. Every track has every registered feature.
    for row in con.execute(
            """SELECT t.track_id, f.name
               FROM tracks t CROSS JOIN features f
               LEFT JOIN curves c ON c.track_id = t.track_id AND c.feature_id = f.feature_id
               WHERE c.track_id IS NULL
               ORDER BY t.track_id, f.feature_id"""):
        problems.append(f"track {row['track_id']}: missing the {row['name']} curve")

    # 2 and 3. Every curve has the right length and no missing values.
    for row in con.execute(
            """SELECT c.track_id, f.name, c.hop_s, c.n_values, c.values_blob, t.duration_s
               FROM curves c
               JOIN tracks t ON t.track_id = c.track_id
               JOIN features f ON f.feature_id = c.feature_id
               ORDER BY c.track_id, f.feature_id"""):
        who = f"track {row['track_id']} {row['name']}"
        if abs(row["hop_s"] - HOP_S) > 1e-9:
            problems.append(f"{who}: step is {row['hop_s']} s, expected {HOP_S} s")
        expected = len(time_grid(row["duration_s"], HOP_S))
        if row["n_values"] != expected:
            problems.append(f"{who}: has {row['n_values']} values, expected {expected} "
                            f"for a {row['duration_s']:.2f} s track")
        if len(row["values_blob"]) != 4 * row["n_values"]:
            problems.append(f"{who}: stored data is damaged (wrong byte count)")
            continue
        values = np.frombuffer(row["values_blob"], dtype="<f4")
        if not np.all(np.isfinite(values)):
            problems.append(f"{who}: contains missing or infinite values")

    # Extra sanity checks on the usable range and the hit points.
    for row in con.execute("SELECT track_id, duration_s, start_s, end_s FROM tracks ORDER BY track_id"):
        if not (0 <= row["start_s"] < row["end_s"] <= row["duration_s"] + 1e-6):
            problems.append(f"track {row['track_id']}: usable range {row['start_s']:.1f}-"
                            f"{row['end_s']:.1f} s does not fit inside {row['duration_s']:.1f} s")
    for row in con.execute(
            """SELECT h.track_id, h.time_s, t.duration_s FROM hit_points h
               JOIN tracks t ON t.track_id = h.track_id
               WHERE h.time_s < 0 OR h.time_s > t.duration_s
               ORDER BY h.track_id, h.time_s"""):
        problems.append(f"track {row['track_id']}: hit at {row['time_s']:.1f} s is outside the track")

    return problems