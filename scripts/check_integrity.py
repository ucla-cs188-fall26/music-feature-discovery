import argparse
import sys

from shapesearch.db import DB_PATH, connect, table_counts
from shapesearch.integrity import find_problems


def main():
    parser = argparse.ArgumentParser(description="Check the Shape Search database.")
    parser.add_argument("--db", default=str(DB_PATH), help="database file (default: %(default)s)")
    args = parser.parse_args()

    con = connect(args.db)
    counts = table_counts(con)
    problems = find_problems(con)
    print(f"Checked {counts['tracks']} tracks, {counts['features']} features, {counts['curves']} curves.")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All checks passed.")
    con.close()
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()