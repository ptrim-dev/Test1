"""Generate a tiny synthetic town (OSM XML) and GTFS feed for tests and demos.

Layout (metres, x east / y north, origin near 51.5N 0.1W):

                 Elm Road (one-way east) y=400
        +-----------------------+----------------+
        |                       |                | Station Road x=800
   North Road               Oak Avenue x=400     |
        |                       |                |
 West --O-- Main Street y=0 ----+----------------+
 Lane   |  (roundabout centred x=-100)           |
   South Road                   |
                                y=-400

Route 10 (shaped): West Lane -> roundabout (2nd exit) -> Main Street ->
left onto Oak Avenue -> right onto Elm Road -> right onto Station Road.
Route 20 (no shapes.txt rows): Main Street east, then right onto Oak Avenue
south; its path is routed between stops through the OSM network.

Usage: python tests/sample_data.py OUT_DIR
"""

from __future__ import annotations

import math
import os
import sys

LAT0, LON0 = 51.5, -0.1
_KY = math.pi / 180.0 * 6371008.8
_KX = _KY * math.cos(math.radians(LAT0))


def ll(x: float, y: float):
    return (LAT0 + y / _KY, LON0 + x / _KX)


RB_C, RB_R = (-100.0, 0.0), 30.0


def ring(angle_deg: float):
    a = math.radians(angle_deg)
    return (RB_C[0] + RB_R * math.cos(a), RB_C[1] + RB_R * math.sin(a))


def build_osm() -> str:
    nodes = {}
    ways = []
    nid = [1000]

    def node(x, y):
        key = (round(x, 3), round(y, 3))
        if key not in nodes:
            nid[0] += 1
            nodes[key] = nid[0]
        return nodes[key]

    def way(points, **tags):
        ways.append(([node(*p) for p in points], tags))

    ring_pts = [ring(a) for a in range(0, 360, 30)]
    way(ring_pts + [ring_pts[0]], highway="primary", junction="roundabout")
    way([(-400, 0), (-250, 0), ring(180)], highway="secondary", name="West Lane")
    way([ring(90), (-100, 200), (-100, 400)], highway="residential", name="North Road")
    way([ring(270), (-100, -200), (-100, -400)], highway="residential", name="South Road")
    way([ring(0), (0, 0), (200, 0), (400, 0)], highway="primary", name="Main Street")
    way([(400, 0), (600, 0), (800, 0)], highway="primary", name="Main Street")
    way([(400, -400), (400, -200), (400, 0), (400, 200), (400, 400)], highway="secondary", name="Oak Avenue")
    way([(-100, 400), (150, 400), (400, 400)], highway="residential", name="Elm Road")
    way([(400, 400), (600, 400), (800, 400)], highway="residential", name="Elm Road", oneway="yes")
    way([(800, 400), (800, 200), (800, 0)], highway="tertiary", name="Station Road")
    # A footpath parallel to Main Street that buses must ignore.
    way([(0, 12), (400, 12)], highway="footway", name="Riverside Walk")

    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="sample_data">']
    for (x, y), i in nodes.items():
        lat, lon = ll(x, y)
        out.append(f'  <node id="{i}" lat="{lat:.7f}" lon="{lon:.7f}"/>')
    for k, (nds, tags) in enumerate(ways, 1):
        out.append(f'  <way id="{k}">')
        out.extend(f'    <nd ref="{n}"/>' for n in nds)
        out.extend(f'    <tag k="{key}" v="{val}"/>' for key, val in tags.items())
        out.append("  </way>")
    out.append("</osm>")
    return "\n".join(out) + "\n"


def route10_shape():
    pts = [(-400, 3), (-250, -2), ring(180)]
    pts += [ring(a) for a in range(210, 361, 30)]
    pts += [(0, 2), (200, -3), (400, 1), (402, 200), (398, 400), (600, 402), (800, 398),
            (801, 200), (799, 5)]
    return pts


STOPS = {
    "S1": ("West Lane West", (-380, 0)),
    "S2": ("Market Square", (150, 0)),
    "S3": ("Oak Avenue Mid", (400, 220)),
    "S4": ("Elm Road East", (650, 400)),
    "S5": ("Station", (800, 50)),
    "S6": ("Main Street East", (700, 0)),
    "S7": ("Oak Avenue South", (400, -350)),
}


def build_gtfs(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def write(name, header, rows):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            fh.write(",".join(header) + "\n")
            for r in rows:
                fh.write(",".join(str(v) for v in r) + "\n")

    write("agency.txt", ["agency_id", "agency_name", "agency_url", "agency_timezone"],
          [["A", "Sample Transit", "https://example.com", "Europe/London"]])
    write("routes.txt", ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"],
          [["R10", "A", "10", "West Lane - Station", 3], ["R20", "A", "20", "Market - Oak South", 3]])
    write("calendar.txt", ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
                           "saturday", "sunday", "start_date", "end_date"],
          [["WK", 1, 1, 1, 1, 1, 0, 0, 20260101, 20261231]])
    write("trips.txt", ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"],
          [["R10", "WK", "T10a", "Station", 0, "SH10"],
           ["R10", "WK", "T10b", "Station", 0, "SH10"],
           ["R20", "WK", "T20a", "Oak Avenue South", 0, ""]])
    write("stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon"],
          [[sid, name, f"{ll(*xy)[0]:.7f}", f"{ll(*xy)[1]:.7f}"] for sid, (name, xy) in STOPS.items()])
    rows = []
    for trip in ("T10a", "T10b"):
        for seq, sid in enumerate(["S1", "S2", "S3", "S4", "S5"], 1):
            t = f"08:{seq * 3:02d}:00"
            rows.append([trip, t, t, sid, seq])
    for seq, sid in enumerate(["S2", "S6", "S7"], 1):
        t = f"09:{seq * 3:02d}:00"
        rows.append(["T20a", t, t, sid, seq])
    write("stop_times.txt", ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], rows)
    write("shapes.txt", ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
          [["SH10", f"{ll(*p)[0]:.7f}", f"{ll(*p)[1]:.7f}", i] for i, p in enumerate(route10_shape(), 1)])


def write_sample(out_dir: str) -> tuple:
    gtfs_dir = os.path.join(out_dir, "gtfs")
    build_gtfs(gtfs_dir)
    osm_path = os.path.join(out_dir, "town.osm")
    with open(osm_path, "w", encoding="utf-8") as fh:
        fh.write(build_osm())
    return gtfs_dir, osm_path


if __name__ == "__main__":
    print(write_sample(sys.argv[1] if len(sys.argv) > 1 else "examples/sample"))
