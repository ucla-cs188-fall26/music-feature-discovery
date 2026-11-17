import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from shapesearch.db import connect, create_schema, table_counts

TABLES = {"curves", "features", "hit_points", "tracks"}
BUILD_SCRIPT = Path(__file__).parent.parent / "scripts" / "build_db.py"


@pytest.fixture
def con(tmp_path):
    con = connect(tmp_path / "test.db")
    create_schema(con)
    yield con
    con.close()


def test_schema_creates_four_empty_tables(con):
    assert table_counts(con) == {t: 0 for t in TABLES}


def test_creating_schema_twice_is_harmless(con):
    create_schema(con)
    assert set(table_counts(con)) == TABLES


def test_curve_for_a_missing_track_is_rejected(con):
    con.execute("INSERT INTO features (name, version, unit, description, method, params_json) "
                "VALUES ('loudness', 'loudness_v1', 'LUFS', 'd', 'm', '{}')")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO curves VALUES (999, 1, 0.1, 1, ?)", (b"\x00\x00\x00\x00",))


def test_license_is_optional_but_title_is_not(con):
    con.execute("INSERT INTO tracks (track_id, title, file_path, duration_s, start_s, end_s) "
                "VALUES (0, 'A', 'a.wav', 60, 0, 60)")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO tracks (track_id, file_path, duration_s, start_s, end_s) "
                    "VALUES (1, 'b.wav', 60, 0, 60)")


def test_feature_names_must_be_unique(con):
    sql = ("INSERT INTO features (name, version, unit, description, method, params_json) "
           "VALUES ('loudness', 'loudness_v1', 'LUFS', 'd', 'm', '{}')")
    con.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(sql)


def test_build_script_twice_then_reset(tmp_path):
    db = tmp_path / "shapes.db"
    for args in ([], [], ["--reset"]):
        done = subprocess.run([sys.executable, str(BUILD_SCRIPT), "--db", str(db), *args],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "Database ready" in done.stdout