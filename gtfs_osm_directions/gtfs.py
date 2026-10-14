"""Minimal GTFS reader: routes, trips, shapes, stops and stop_times."""

from __future__ import annotations

import csv
import io
import os
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

# GTFS route_type -> travel mode used to pick OSM ways.
ROUTE_TYPE_MODES = {
    0: "tram", 1: "rail", 2: "rail", 3: "bus", 4: "ferry", 5: "tram",
    6: "aerial", 7: "rail", 11: "bus", 12: "rail",
}


def route_type_mode(route_type: int) -> str:
    if route_type in ROUTE_TYPE_MODES:
        return ROUTE_TYPE_MODES[route_type]
    # Extended (Google) route types: 100-199 rail, 200-299 coach, 400-499
    # urban rail, 700-799 bus, 800 trolleybus, 900-999 tram.
    hundreds = route_type // 100
    return {1: "rail", 2: "bus", 4: "rail", 7: "bus", 8: "bus", 9: "tram"}.get(hundreds, "bus")


@dataclass
class Route:
    route_id: str
    short_name: str
    long_name: str
    route_type: int

    @property
    def mode(self) -> str:
        return route_type_mode(self.route_type)

    @property
    def label(self) -> str:
        if self.short_name and self.long_name:
            return f"{self.short_name} – {self.long_name}"
        return self.short_name or self.long_name or self.route_id


@dataclass
class Stop:
    stop_id: str
    name: str
    lat: float
    lon: float


@dataclass
class Trip:
    trip_id: str
    route_id: str
    shape_id: Optional[str]
    direction_id: Optional[str]
    headsign: str


@dataclass
class Pattern:
    """A distinct path a route runs: one representative trip plus its siblings."""

    route: Route
    direction_id: Optional[str]
    shape_id: Optional[str]
    headsign: str
    trip_ids: List[str]
    stop_ids: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        parts = [self.route.route_id, self.direction_id or "-", self.shape_id or "noshape"]
        return ":".join(parts)


def _open_tables(path: str):
    if os.path.isdir(path):
        def opener(name: str):
            full = os.path.join(path, name)
            if not os.path.exists(full):
                return None
            return open(full, newline="", encoding="utf-8-sig")
        return opener
    zf = zipfile.ZipFile(path)
    names = {os.path.basename(n): n for n in zf.namelist() if not n.endswith("/")}

    def opener(name: str):
        if name not in names:
            return None
        return io.TextIOWrapper(zf.open(names[name]), encoding="utf-8-sig", newline="")
    return opener


def _rows(opener, name: str, required: bool = True) -> Iterator[Dict[str, str]]:
    fh = opener(name)
    if fh is None:
        if required:
            raise FileNotFoundError(f"GTFS feed is missing {name}")
        return
    with fh:
        for row in csv.DictReader(fh):
            yield {(k or "").strip(): (v or "").strip() for k, v in row.items()}


class GTFSFeed:
    def __init__(self, path: str):
        opener = _open_tables(path)
        self.routes: Dict[str, Route] = {}
        for r in _rows(opener, "routes.txt"):
            self.routes[r["route_id"]] = Route(
                r["route_id"], r.get("route_short_name", ""),
                r.get("route_long_name", ""), int(r.get("route_type") or 3),
            )
        self.stops: Dict[str, Stop] = {}
        for r in _rows(opener, "stops.txt"):
            if not r.get("stop_lat") or not r.get("stop_lon"):
                continue
            self.stops[r["stop_id"]] = Stop(
                r["stop_id"], r.get("stop_name", ""), float(r["stop_lat"]), float(r["stop_lon"])
            )
        self.trips: Dict[str, Trip] = {}
        for r in _rows(opener, "trips.txt"):
            self.trips[r["trip_id"]] = Trip(
                r["trip_id"], r["route_id"], r.get("shape_id") or None,
                r.get("direction_id") or None, r.get("trip_headsign", ""),
            )
        shape_pts: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
        for r in _rows(opener, "shapes.txt", required=False):
            shape_pts[r["shape_id"]].append(
                (int(float(r["shape_pt_sequence"])), float(r["shape_pt_lat"]), float(r["shape_pt_lon"]))
            )
        self.shapes: Dict[str, List[Tuple[float, float]]] = {
            sid: [(lat, lon) for _, lat, lon in sorted(pts)] for sid, pts in shape_pts.items()
        }
        seqs: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for r in _rows(opener, "stop_times.txt"):
            if r["trip_id"] in self.trips:
                seqs[r["trip_id"]].append((int(float(r["stop_sequence"])), r["stop_id"]))
        self.trip_stops: Dict[str, List[str]] = {
            tid: [sid for _, sid in sorted(v)] for tid, v in seqs.items()
        }

    def patterns(self, route_filter: Optional[List[str]] = None) -> List[Pattern]:
        """Group trips into distinct paths (shape, or stop sequence when unshaped)."""
        wanted = None
        if route_filter:
            wanted = set(route_filter)
        groups: Dict[tuple, List[str]] = defaultdict(list)
        for trip in self.trips.values():
            route = self.routes.get(trip.route_id)
            if route is None:
                continue
            if wanted and not ({route.route_id, route.short_name} & wanted):
                continue
            stops = tuple(self.trip_stops.get(trip.trip_id, ()))
            path_key = trip.shape_id if trip.shape_id in self.shapes else stops
            groups[(trip.route_id, trip.direction_id, path_key)].append(trip.trip_id)

        out = []
        for (route_id, direction_id, _), trip_ids in groups.items():
            # Representative trip: the most common stop pattern within the group.
            counts = Counter(tuple(self.trip_stops.get(t, ())) for t in trip_ids)
            stop_ids = list(counts.most_common(1)[0][0])
            rep = next(t for t in trip_ids if tuple(self.trip_stops.get(t, ())) == tuple(stop_ids))
            trip = self.trips[rep]
            shape_id = trip.shape_id if trip.shape_id in self.shapes else None
            headsign = Counter(self.trips[t].headsign for t in trip_ids).most_common(1)[0][0]
            out.append(Pattern(self.routes[route_id], direction_id, shape_id, headsign,
                               sorted(trip_ids), stop_ids))
        out.sort(key=lambda p: (p.route.short_name or p.route.route_id, p.direction_id or "",
                                -len(p.trip_ids)))
        return out
