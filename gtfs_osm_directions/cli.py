"""Command line interface."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .directions import DirectionParams
from .gtfs import GTFSFeed
from .matcher import MatchParams
from .osm import fetch_overpass, load_osm
from .output import FORMATS
from .pipeline import DirectionsEngine, feed_bbox, feed_projection


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gtfs-osm-directions",
        description="Generate turn-by-turn directions for GTFS route paths using OpenStreetMap data.",
    )
    ap.add_argument("gtfs", help="GTFS feed (.zip or directory)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--osm", help="OSM extract (.osm, .osm.gz, .osm.bz2, or .osm.pbf with pyosmium)")
    src.add_argument("--overpass", action="store_true",
                     help="download the needed ways from the Overpass API for the routes' bounding box")
    ap.add_argument("--overpass-cache", default="overpass_cache.osm",
                    help="where to store/reuse the Overpass download (default: %(default)s)")
    ap.add_argument("--overpass-url", default=None, help="alternative Overpass endpoint")
    ap.add_argument("-r", "--route", action="append", dest="routes",
                    help="route_id or route_short_name to process (repeatable; default: all)")
    ap.add_argument("--direction", choices=["0", "1"], help="only this direction_id")
    ap.add_argument("-f", "--format", choices=sorted(FORMATS), default="text")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    ap.add_argument("--step", type=float, default=MatchParams.step, help="shape resample spacing in metres")
    ap.add_argument("--radius", type=float, default=MatchParams.radius, help="max distance from shape to OSM way (m)")
    ap.add_argument("--straight-threshold", type=float, default=DirectionParams.straight,
                    help="turns below this angle are 'continue' (degrees)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    feed = GTFSFeed(args.gtfs)
    patterns = feed.patterns(args.routes)
    if args.direction is not None:
        patterns = [p for p in patterns if p.direction_id == args.direction]
    if not patterns:
        print("No matching routes/trips found in the GTFS feed.", file=sys.stderr)
        return 1

    if args.overpass:
        if os.path.exists(args.overpass_cache):
            print(f"Using cached Overpass data {args.overpass_cache}", file=sys.stderr)
        else:
            bbox = feed_bbox(feed, patterns)
            modes = {p.route.mode for p in patterns}
            print(f"Downloading OSM ways for bbox {bbox} from Overpass…", file=sys.stderr)
            kwargs = {"url": args.overpass_url} if args.overpass_url else {}
            fetch_overpass(bbox, modes, args.overpass_cache, **kwargs)
        osm_path = args.overpass_cache
    else:
        osm_path = args.osm
    osm = load_osm(osm_path)

    engine = DirectionsEngine(
        feed, osm, feed_projection(feed, patterns),
        MatchParams(step=args.step, radius=args.radius),
        DirectionParams(straight=args.straight_threshold),
    )
    results = []
    for i, p in enumerate(patterns, 1):
        print(f"[{i}/{len(patterns)}] route {p.route.label} {p.key}", file=sys.stderr)
        results.append(engine.run(p))

    rendered = FORMATS[args.format](results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered if rendered.endswith("\n") else rendered + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
