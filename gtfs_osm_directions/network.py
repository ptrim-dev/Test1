"""A projected, spatially indexed OSM network for one travel mode."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .geo import XY, LocalProjection, bearing_xy, dist, project_to_segment
from .osm import OSMData, Way

BUS_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road", "busway", "bus_guideway",
}
TRAM_RAILWAYS = {"tram", "light_rail"}
RAIL_RAILWAYS = {"rail", "subway", "light_rail", "narrow_gauge", "monorail", "funicular"}
_YES = {"yes", "designated", "permissive"}
_NO = {"no", "private"}


def way_allowed(tags: Dict[str, str], mode: str) -> bool:
    if mode == "tram":
        return tags.get("railway") in TRAM_RAILWAYS
    if mode == "rail":
        return tags.get("railway") in RAIL_RAILWAYS
    hw = tags.get("highway")
    bus_ok = tags.get("bus") in _YES or tags.get("psv") in _YES
    if hw == "pedestrian" or hw == "footway":
        return bus_ok
    if hw not in BUS_HIGHWAYS:
        return False
    if hw == "service" and tags.get("service") in {"parking_aisle", "driveway", "drive-through"}:
        return bus_ok
    if (tags.get("access") in _NO or tags.get("motor_vehicle") in _NO) and not bus_ok:
        return False
    return True


def way_direction(tags: Dict[str, str], mode: str) -> int:
    """0 = both directions, 1 = only along node order, -1 = only against it."""
    if mode != "bus":
        return 0
    for key in ("oneway:bus", "oneway:psv"):
        if tags.get(key) == "no":
            return 0
    if tags.get("busway") in {"opposite_lane", "opposite_track"}:
        return 0
    ow = tags.get("oneway")
    if ow in {"yes", "true", "1"}:
        return 1
    if ow in {"-1", "reverse"}:
        return -1
    if ow == "no":
        return 0
    if tags.get("junction") in {"roundabout", "circular"} or tags.get("highway") == "motorway":
        return 1
    return 0


def is_roundabout(tags: Dict[str, str]) -> bool:
    return tags.get("junction") in {"roundabout", "circular"}


@dataclass
class Segment:
    way_id: int
    index: int  # segment between way.node_ids[index] and [index + 1]
    a: XY
    b: XY
    bearing: float
    length: float


class Network:
    def __init__(self, osm: OSMData, mode: str, projection: LocalProjection, cell: float = 100.0):
        self.mode = mode
        self.proj = projection
        self.cell = cell
        self.ways: Dict[int, Way] = {}
        self.direction: Dict[int, int] = {}
        self.node_xy: Dict[int, XY] = {}
        self.node_ways: Dict[int, Set[int]] = defaultdict(set)
        self.segments: List[Segment] = []
        self.way_segments: Dict[int, List[int]] = defaultdict(list)
        self.grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for way in osm.ways.values():
            if not way_allowed(way.tags, mode):
                continue
            self.ways[way.way_id] = way
            self.direction[way.way_id] = way_direction(way.tags, mode)
            for n in way.node_ids:
                if n not in self.node_xy:
                    self.node_xy[n] = projection.to_xy(*osm.nodes[n])
                self.node_ways[n].add(way.way_id)
            for i, (n1, n2) in enumerate(zip(way.node_ids, way.node_ids[1:])):
                a, b = self.node_xy[n1], self.node_xy[n2]
                sid = len(self.segments)
                self.segments.append(Segment(way.way_id, i, a, b, bearing_xy(a, b), dist(a, b)))
                self.way_segments[way.way_id].append(sid)
                for key in self._cells(a, b):
                    self.grid[key].append(sid)
        self._adjacency: Optional[Dict[int, List[Tuple[int, float, int]]]] = None
        self._connected: Dict[Tuple[int, int], bool] = {}

    def __len__(self) -> int:
        return len(self.ways)

    def _cells(self, a: XY, b: XY) -> Iterable[Tuple[int, int]]:
        c = self.cell
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        for cx in range(math.floor(x0 / c), math.floor(x1 / c) + 1):
            for cy in range(math.floor(y0 / c), math.floor(y1 / c) + 1):
                yield (cx, cy)

    def nearby_segments(self, p: XY, radius: float) -> List[Tuple[int, float, float, XY]]:
        """Segments within radius of p -> [(segment id, distance, t, projected point)]."""
        c = self.cell
        seen: Set[int] = set()
        out = []
        for cx in range(math.floor((p[0] - radius) / c), math.floor((p[0] + radius) / c) + 1):
            for cy in range(math.floor((p[1] - radius) / c), math.floor((p[1] + radius) / c) + 1):
                for sid in self.grid.get((cx, cy), ()):
                    if sid in seen:
                        continue
                    seen.add(sid)
                    seg = self.segments[sid]
                    d, t, q = project_to_segment(p, seg.a, seg.b)
                    if d <= radius:
                        out.append((sid, d, t, q))
        return out

    def connected(self, w1: int, w2: int) -> bool:
        if w1 == w2:
            return True
        key = (w1, w2) if w1 < w2 else (w2, w1)
        hit = self._connected.get(key)
        if hit is None:
            hit = not set(self.ways[w1].node_ids).isdisjoint(self.ways[w2].node_ids)
            self._connected[key] = hit
        return hit

    def label(self, way_id: int) -> Optional[str]:
        tags = self.ways[way_id].tags
        name, ref = tags.get("name"), tags.get("ref")
        if name and ref:
            return f"{name} ({ref})"
        return name or ref

    # ---------------------------------------------------------------- routing
    def adjacency(self) -> Dict[int, List[Tuple[int, float, int]]]:
        if self._adjacency is None:
            adj: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
            for wid, way in self.ways.items():
                d = self.direction[wid]
                for n1, n2 in zip(way.node_ids, way.node_ids[1:]):
                    length = dist(self.node_xy[n1], self.node_xy[n2])
                    if d >= 0:
                        adj[n1].append((n2, length, wid))
                    if d <= 0:
                        adj[n2].append((n1, length, wid))
            self._adjacency = adj
        return self._adjacency

    def nearest_segment(self, p: XY, max_radius: float = 500.0):
        radius = 50.0
        while radius <= max_radius:
            hits = self.nearby_segments(p, radius)
            if hits:
                return min(hits, key=lambda h: h[1])
            radius *= 2
        return None

    def route(self, start: XY, end: XY, max_radius: float = 500.0) -> Optional[List[XY]]:
        """Shortest path between two arbitrary points via the network (A*)."""
        s_hit = self.nearest_segment(start, max_radius)
        e_hit = self.nearest_segment(end, max_radius)
        if s_hit is None or e_hit is None:
            return None
        s_seg, e_seg = self.segments[s_hit[0]], self.segments[e_hit[0]]
        if s_hit[0] == e_hit[0] and e_hit[2] >= s_hit[2] and self.direction[s_seg.way_id] >= 0:
            return [s_hit[3], e_hit[3]]
        if s_hit[0] == e_hit[0] and e_hit[2] <= s_hit[2] and self.direction[s_seg.way_id] <= 0:
            return [s_hit[3], e_hit[3]]

        s_way, e_way = self.ways[s_seg.way_id], self.ways[e_seg.way_id]
        s_n1, s_n2 = s_way.node_ids[s_seg.index], s_way.node_ids[s_seg.index + 1]
        e_n1, e_n2 = e_way.node_ids[e_seg.index], e_way.node_ids[e_seg.index + 1]
        sd, ed = self.direction[s_seg.way_id], self.direction[e_seg.way_id]

        starts = []  # (node, cost to reach it from the start point)
        if sd <= 0:
            starts.append((s_n1, s_hit[2] * s_seg.length))
        if sd >= 0:
            starts.append((s_n2, (1 - s_hit[2]) * s_seg.length))
        goals = {}  # node -> remaining cost to the end point
        if ed >= 0:
            goals[e_n1] = e_hit[2] * e_seg.length
        if ed <= 0:
            goals[e_n2] = min(goals.get(e_n2, math.inf), (1 - e_hit[2]) * e_seg.length)

        adj = self.adjacency()
        xy = self.node_xy
        best: Dict[int, float] = {}
        prev: Dict[int, Optional[int]] = {}
        heap = []
        for n, c in starts:
            if c < best.get(n, math.inf):
                best[n] = c
                prev[n] = None
                heapq.heappush(heap, (c + dist(xy[n], end), c, n))
        found, found_cost = None, math.inf
        while heap:
            f, g, n = heapq.heappop(heap)
            if g > best.get(n, math.inf) or f >= found_cost:
                continue
            if n in goals and g + goals[n] < found_cost:
                found, found_cost = n, g + goals[n]
            for m, length, _ in adj.get(n, ()):
                ng = g + length
                if ng < best.get(m, math.inf):
                    best[m] = ng
                    prev[m] = n
                    heapq.heappush(heap, (ng + dist(xy[m], end), ng, m))
        if found is None:
            return None
        path = []
        n: Optional[int] = found
        while n is not None:
            path.append(xy[n])
            n = prev[n]
        path.reverse()
        return [s_hit[3]] + path + [e_hit[3]]
