"""End-to-end: GTFS pattern + OSM network -> directions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .directions import DirectionBuilder, DirectionParams, Instruction
from .geo import XY, LocalProjection, cumulative_lengths, resample
from .gtfs import GTFSFeed, Pattern
from .matcher import MatchParams, match
from .network import Network
from .osm import OSMData

SUPPORTED_MODES = {"bus", "tram", "rail"}


@dataclass
class PatternDirections:
    pattern: Pattern
    instructions: List[Instruction] = field(default_factory=list)
    geometry: List[Tuple[float, float]] = field(default_factory=list)  # (lat, lon)
    length_m: float = 0.0
    matched_ratio: float = 0.0
    source: str = "shape"  # shape | routed-from-stops
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        p = self.pattern
        return {
            "route_id": p.route.route_id,
            "route_short_name": p.route.short_name,
            "route_long_name": p.route.long_name,
            "mode": p.route.mode,
            "direction_id": p.direction_id,
            "shape_id": p.shape_id,
            "headsign": p.headsign,
            "trip_count": len(p.trip_ids),
            "path_source": self.source,
            "length_m": round(self.length_m, 1),
            "matched_ratio": round(self.matched_ratio, 3),
            "warnings": self.warnings,
            "instructions": [i.to_dict() for i in self.instructions],
        }


def feed_projection(feed: GTFSFeed, patterns: Iterable[Pattern]) -> LocalProjection:
    coords: List[Tuple[float, float]] = []
    for p in patterns:
        if p.shape_id:
            coords.extend(feed.shapes[p.shape_id])
        coords.extend((feed.stops[s].lat, feed.stops[s].lon) for s in p.stop_ids if s in feed.stops)
    return LocalProjection.around(coords)


def feed_bbox(feed: GTFSFeed, patterns: Iterable[Pattern], buffer_deg: float = 0.005):
    lats, lons = [], []
    for p in patterns:
        pts = list(feed.shapes.get(p.shape_id, [])) if p.shape_id else []
        pts += [(feed.stops[s].lat, feed.stops[s].lon) for s in p.stop_ids if s in feed.stops]
        for lat, lon in pts:
            lats.append(lat)
            lons.append(lon)
    if not lats:
        raise ValueError("selected patterns have no coordinates")
    return (min(lats) - buffer_deg, min(lons) - buffer_deg,
            max(lats) + buffer_deg, max(lons) + buffer_deg)


class DirectionsEngine:
    def __init__(self, feed: GTFSFeed, osm: OSMData, projection: LocalProjection,
                 match_params: MatchParams = MatchParams(),
                 direction_params: DirectionParams = DirectionParams()):
        self.feed = feed
        self.osm = osm
        self.proj = projection
        self.match_params = match_params
        self.direction_params = direction_params
        self._networks: Dict[str, Network] = {}

    def network(self, mode: str) -> Network:
        if mode not in self._networks:
            self._networks[mode] = Network(self.osm, mode, self.proj)
        return self._networks[mode]

    def _path_from_stops(self, net: Network, stops: List[XY], warnings: List[str]) -> List[XY]:
        path: List[XY] = []
        for i, (a, b) in enumerate(zip(stops, stops[1:])):
            leg = net.route(a, b)
            if leg is None:
                warnings.append(f"no network path between stops {i + 1} and {i + 2}; using a straight line")
                leg = [a, b]
            path.extend(leg if not path else leg[1:])
        return path

    def run(self, pattern: Pattern) -> PatternDirections:
        result = PatternDirections(pattern)
        mode = pattern.route.mode
        if mode not in SUPPORTED_MODES:
            result.warnings.append(f"mode '{mode}' has no OSM street network; skipped")
            return result
        net = self.network(mode)
        if not len(net):
            result.warnings.append(f"OSM data contains no ways usable by mode '{mode}'")
            return result

        stops = [self.feed.stops[s] for s in pattern.stop_ids if s in self.feed.stops]
        stop_xy = [self.proj.to_xy(s.lat, s.lon) for s in stops]
        if pattern.shape_id:
            raw = [self.proj.to_xy(lat, lon) for lat, lon in self.feed.shapes[pattern.shape_id]]
        else:
            result.source = "routed-from-stops"
            raw = self._path_from_stops(net, stop_xy, result.warnings)
        if len(raw) < 2:
            result.warnings.append("pattern has fewer than two points")
            return result

        samples = resample(raw, self.match_params.step)
        matched = match(samples, net, self.match_params)
        result.matched_ratio = sum(1 for m in matched if m.way_id is not None) / len(matched)
        if result.matched_ratio < 0.8:
            result.warnings.append(
                f"only {result.matched_ratio:.0%} of the path matched OSM ways; "
                "check that the OSM extract covers the route")
        line = [m.snapped or m.xy for m in matched]
        result.length_m = cumulative_lengths(line)[-1]
        result.geometry = [self.proj.to_latlon(*p) for p in line]

        destination = stops[-1].name if stops else (pattern.headsign or None)
        builder = DirectionBuilder(net, self.direction_params)
        result.instructions = builder.build(
            matched, [(s.stop_id, s.name, xy) for s, xy in zip(stops, stop_xy)], destination)
        return result
