"""
Download real Dwarka (Delhi) roads and hospitals from OpenStreetMap and save
them to data/dwarka_graph.json for the app's "Dwarka (real map)" mode.

Usage (needs internet access and `pip install requests`):
    python scripts/fetch_dwarka_osm.py                # arterial + sector roads
    python scripts/fetch_dwarka_osm.py --detail all   # also residential streets
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.osm_builder import fetch_and_build, DEFAULT_OUTPUT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", choices=["major", "all"], default="major",
                        help="'major' = primary/secondary/tertiary roads (default); "
                             "'all' adds residential streets (bigger, slower).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print("Downloading Dwarka from OpenStreetMap (can take up to a minute)...")
    try:
        meta = fetch_and_build(args.output, detail=args.detail)
    except (RuntimeError, ValueError) as exc:
        print(f"\nFailed: {exc}", file=sys.stderr)
        print("Check your internet connection (openstreetmap/overpass must be "
              "reachable) and try again.", file=sys.stderr)
        sys.exit(1)
    print(f"Saved {args.output}")
    print(f"  intersections: {meta['node_count']}  roads: {meta['edge_count']}  "
          f"hospitals: {meta['hospital_count']}")


if __name__ == "__main__":
    main()
