import argparse
import sys
from pathlib import Path

from shapesearch.db import DB_PATH, connect, create_schema, table_counts
from shapesearch.features.extract import MANIFEST, OUT_DIR
from shapesearch.ingest import load_catalog


def main():
    parser = argparse.ArgumentParser(description="Create the Shape Search database.")
    parser.add_argument("--db", default=str(DB_PATH), help="database file (default: %(default)s)")
    parser.add_argument("--reset", action="store_true", help="delete the database first")
    parser.add_argument("--load", action="store_true",
                        help="load every track in the manifest from the extracted features")
    parser.add_argument("--manifest", default=str(MANIFEST), help="manifest CSV (default: %(default)s)")
    parser.add_argument("--features", default=str(OUT_DIR), help="folder of extracted features (default: %(default)s)")
    args = parser.parse_args()

    db = Path(args.db)
    if args.reset and db.exists():
        db.unlink()
        print(f"Deleted {db}")

    con = connect(db)
    create_schema(con)
    print(f"Database ready: {db}")
    failed = []
    if args.load:
        result = load_catalog(con, args.manifest, args.features)
        failed = result["failed"]
        print(f"Loaded {result['loaded']} tracks.")
        for track_id, why in failed:
            print(f"  track {track_id:04d} NOT loaded: {why}")
    for table, n in table_counts(con).items():
        print(f"  {table:<12}{n:>6} rows")
    con.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
