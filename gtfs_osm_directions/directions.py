"""Turn a matched path into turn-by-turn instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .geo import XY, angle_diff, bearing_xy, cardinal, cumulative_lengths, dist, project_to_segment
from .matcher import MatchedPoint
from .network import Network, is_roundabout


@dataclass
class DirectionParams:
    bearing_window: float = 30.0  # metres before/after a junction used for angles
    straight: float = 20.0
    slight: float = 45.0
    turn: float = 135.0
    sharp: float = 170.0
    min_leg: float = 40.0  # A-B-A legs shorter than this are absorbed
    min_offnet_gap: float = 60.0  # unmatched stretches shorter than this are absorbed


@dataclass
class StopRef:
    stop_id: str
    name: str
    distance: float  # metres from route start


@dataclass
class Instruction:
    action: str  # depart, continue, slight, turn, sharp, uturn, ramp, merge, roundabout, offnet, arrive
    modifier: Optional[str]  # left / right / straight
    street: Optional[str]
    text: str
    lat: float
    lon: float
    start_distance: float
    distance: float  # metres travelled after this instruction until the next one
    turn_angle: Optional[float] = None
    roundabout_exit: Optional[int] = None
    stops: List[StopRef] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "modifier": self.modifier,
            "street": self.street,
            "text": self.text,
            "location": [round(self.lat, 7), round(self.lon, 7)],
            "start_distance_m": round(self.start_distance, 1),
            "distance_m": round(self.distance, 1),
            "turn_angle": None if self.turn_angle is None else round(self.turn_angle, 1),
            "roundabout_exit": self.roundabout_exit,
            "stops": [{"stop_id": s.stop_id, "name": s.name, "distance_m": round(s.distance, 1)}
                      for s in self.stops],
        }


@dataclass
class _Leg:
    start: int  # first sample index
    end: int  # first sample index of the next leg (exclusive)
    key: tuple
    way_ids: List[int]


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def format_distance(m: float) -> str:
    if m >= 1000:
        return f"{m / 1000:.1f} km"
    if m >= 100:
        return f"{int(round(m / 10.0) * 10)} m"
    return f"{int(round(m))} m"


class DirectionBuilder:
    def __init__(self, net: Network, params: DirectionParams = DirectionParams()):
        self.net = net
        self.params = params

    # ------------------------------------------------------------------ legs
    def _key(self, way_id: Optional[int]) -> tuple:
        if way_id is None:
            return ("offnet",)
        tags = self.net.ways[way_id].tags
        if is_roundabout(tags):
            return ("roundabout",)
        label = self.net.label(way_id)
        if label:
            return ("street", label)
        return ("unnamed", tags.get("highway") or tags.get("railway") or "road")

    def _legs(self, pts: Sequence[MatchedPoint], cum: List[float]) -> List[_Leg]:
        ways = [p.way_id for p in pts]
        n = len(ways)
        # Absorb short unmatched gaps into the preceding (or following) way.
        i = 0
        while i < n:
            if ways[i] is None:
                j = i
                while j < n and ways[j] is None:
                    j += 1
                span = cum[min(j, n - 1)] - cum[max(i - 1, 0)]
                if span < self.params.min_offnet_gap and (i > 0 or j < n):
                    fill = ways[i - 1] if i > 0 else ways[j]
                    for k in range(i, j):
                        ways[k] = fill
                i = j
            else:
                i += 1

        legs: List[_Leg] = []
        for i, w in enumerate(ways):
            key = self._key(w)
            reversed_here = (i > 0 and w is not None and w == ways[i - 1]
                             and pts[i].forward != pts[i - 1].forward)
            if legs and legs[-1].key == key and not reversed_here:
                legs[-1].end = i + 1
                if w is not None and legs[-1].way_ids[-1] != w:
                    legs[-1].way_ids.append(w)
            else:
                legs.append(_Leg(i, i + 1, key, [w] if w is not None else []))

        # Absorb A-B-A blips (e.g. briefly snapping to a cross street).
        changed = True
        while changed:
            changed = False
            for k in range(1, len(legs) - 1):
                a, b, c = legs[k - 1], legs[k], legs[k + 1]
                length = cum[min(b.end, n - 1)] - cum[b.start]
                if a.key == c.key and b.key[0] != "roundabout" and length < self.params.min_leg:
                    a.end = c.end
                    a.way_ids += [w for w in c.way_ids if w not in a.way_ids]
                    del legs[k:k + 2]
                    changed = True
                    break
        return legs

    # ---------------------------------------------------------------- angles
    def _bearing(self, pts, cum, frm: int, to: int) -> float:
        a = pts[frm].snapped or pts[frm].xy
        b = pts[to].snapped or pts[to].xy
        if a == b:
            return pts[to].heading
        return bearing_xy(a, b)

    def _index_at(self, cum, target: float, lo: int, hi: int) -> int:
        i = lo
        if target >= cum[lo]:
            while i < hi and cum[i] < target:
                i += 1
        else:
            while i > hi and cum[i] > target:
                i -= 1
        return i

    def _turn_angle(self, pts, cum, prev: _Leg, nxt_start: int, nxt_end: int) -> float:
        w = self.params.bearing_window
        b = nxt_start
        before = self._index_at(cum, cum[b] - w, b, prev.start)
        after = self._index_at(cum, cum[b] + w, b, max(b, nxt_end - 1))
        if before == b:
            before = max(prev.start, b - 1)
        if after == b:
            after = min(len(pts) - 1, b + 1)
        return angle_diff(self._bearing(pts, cum, before, b), self._bearing(pts, cum, b, after))

    def _classify(self, angle: float) -> Tuple[str, str]:
        p = self.params
        side = "right" if angle > 0 else "left"
        a = abs(angle)
        if a <= p.straight:
            return "continue", "straight"
        if a <= p.slight:
            return "slight", side
        if a <= p.turn:
            return "turn", side
        if a <= p.sharp:
            return "sharp", side
        return "uturn", None

    # ------------------------------------------------------------ roundabouts
    def _shared_node(self, ways_a: List[int], ways_b: List[int]) -> Optional[int]:
        nodes_b = set()
        for w in ways_b:
            nodes_b.update(self.net.ways[w].node_ids)
        for w in ways_a:
            for n in self.net.ways[w].node_ids:
                if n in nodes_b:
                    return n
        return None

    def _nearest_node(self, way_ids: List[int], p: XY) -> Optional[int]:
        best, best_d = None, float("inf")
        for w in way_ids:
            for n in self.net.ways[w].node_ids:
                d = dist(self.net.node_xy[n], p)
                if d < best_d:
                    best, best_d = n, d
        return best

    def _can_leave(self, way_id: int, node: int) -> bool:
        ids = self.net.ways[way_id].node_ids
        d = self.net.direction[way_id]
        if d == 0:
            return True
        positions = [i for i, n in enumerate(ids) if n == node]
        if d == 1:
            return any(i < len(ids) - 1 for i in positions)
        return any(i > 0 for i in positions)

    def roundabout_exit(self, ring: List[int], entry: int, exit_: int) -> Optional[int]:
        succ: Dict[int, int] = {}
        for w in ring:
            ids = self.net.ways[w].node_ids
            if self.net.direction[w] == -1:
                ids = ids[::-1]
            for a, b in zip(ids, ids[1:]):
                succ[a] = b
        ring_set = set(ring)
        count, cur = 0, entry
        for _ in range(10000):
            if cur not in succ:
                return None
            cur = succ[cur]
            if any(w not in ring_set and not is_roundabout(self.net.ways[w].tags)
                   and self._can_leave(w, cur) for w in self.net.node_ways[cur]):
                count += 1
            if cur == exit_:
                return max(count, 1)
            if cur == entry:
                return None
        return None

    # ----------------------------------------------------------------- build
    def build(self, pts: Sequence[MatchedPoint], stops: Sequence[Tuple[str, str, XY]] = (),
              destination: Optional[str] = None) -> List[Instruction]:
        if not pts:
            return []
        # Measure along the snapped road geometry so shape noise doesn't inflate distances.
        line = [p.snapped or p.xy for p in pts]
        cum = cumulative_lengths(line)
        total = cum[-1]
        legs = self._legs(pts, cum)
        proj = self.net.proj
        out: List[Instruction] = []

        def leg_label(leg: _Leg) -> Optional[str]:
            if leg.key[0] == "street":
                return leg.key[1]
            if leg.key[0] == "unnamed":
                return f"unnamed {leg.key[1].replace('_', ' ')}"
            return None

        def emit(action, modifier, street, text, idx, end_idx, angle=None, exit_no=None):
            lat, lon = proj.to_latlon(*(pts[idx].snapped or pts[idx].xy))
            start = cum[idx]
            end = cum[end_idx] if end_idx < len(cum) else total
            out.append(Instruction(action, modifier, street, text, lat, lon, start,
                                   max(0.0, end - start), angle, exit_no))

        k = 0
        while k < len(legs):
            leg = legs[k]
            label = leg_label(leg)
            if k == 0:
                heading = cardinal(self._bearing(pts, cum, leg.start, min(len(pts) - 1, leg.start + 3)))
                if leg.key[0] == "offnet":
                    text = f"Head {heading} (no matching OSM way)"
                elif leg.key[0] == "roundabout":
                    text = f"Head {heading} on the roundabout"
                else:
                    text = f"Head {heading} on {label}"
                emit("depart", None, label, text, leg.start, leg.end)
                k += 1
                continue

            prev = legs[k - 1]
            if leg.key[0] == "roundabout" and k + 1 < len(legs):
                nxt = legs[k + 1]
                nlabel = leg_label(nxt)
                entry = (self._shared_node(prev.way_ids, leg.way_ids) if prev.way_ids else None) \
                    or self._nearest_node(leg.way_ids, pts[leg.start].xy)
                exit_ = (self._shared_node(nxt.way_ids, leg.way_ids) if nxt.way_ids else None) \
                    or self._nearest_node(leg.way_ids, pts[leg.end - 1].xy)
                exit_no = self.roundabout_exit(leg.way_ids, entry, exit_) if entry and exit_ else None
                onto = f" onto {nlabel}" if nlabel else ""
                if exit_no:
                    text = f"At the roundabout, take the {ordinal(exit_no)} exit{onto}"
                else:
                    text = f"Enter the roundabout and exit{onto}"
                emit("roundabout", None, nlabel, text, leg.start, nxt.end, exit_no=exit_no)
                k += 2
                continue

            if leg.key[0] == "offnet":
                emit("offnet", None, None, "Continue (no matching OSM way)", leg.start, leg.end)
                k += 1
                continue
            if leg.key[0] == "roundabout":  # roundabout at the very end of the route
                emit("roundabout", None, None, "Enter the roundabout", leg.start, leg.end)
                k += 1
                continue

            angle = self._turn_angle(pts, cum, prev, leg.start, leg.end)
            action, modifier = self._classify(angle)
            onto = f" onto {label}" if label else ""
            hw_next = self.net.ways[leg.way_ids[0]].tags.get("highway", "")
            hw_prev = self.net.ways[prev.way_ids[-1]].tags.get("highway", "") if prev.way_ids else ""
            if hw_next.endswith("_link") and not hw_prev.endswith("_link"):
                action = "ramp"
                side = "" if modifier == "straight" else f" on the {modifier}"
                text = f"Take the ramp{side}{onto}"
            elif hw_prev.endswith("_link") and not hw_next.endswith("_link") and action in ("continue", "slight"):
                action = "merge"
                text = f"Merge{onto}"
            elif action == "continue":
                text = f"Continue straight{onto}"
            elif action == "slight":
                text = f"Bear {modifier}{onto}"
            elif action == "turn":
                text = f"Turn {modifier}{onto}"
            elif action == "sharp":
                text = f"Make a sharp {modifier}{onto}"
            elif label and label == leg_label(prev):
                text = f"Make a U-turn to stay on {label}"
            else:
                text = f"Make a U-turn{onto}"
            emit(action, modifier, label, text, leg.start, leg.end, angle=angle)
            k += 1

        last = pts[-1].snapped or pts[-1].xy
        lat, lon = proj.to_latlon(*last)
        dest = f"Arrive at {destination}" if destination else "Arrive at your destination"
        out.append(Instruction("arrive", None, None, dest, lat, lon, total, 0.0))

        prev_d = -1.0
        for stop_id, name, sxy in stops:
            d = locate_along(line, cum, sxy, after=prev_d)
            prev_d = d
            for ins in out:
                if ins.start_distance <= d < ins.start_distance + ins.distance or ins is out[-1]:
                    ins.stops.append(StopRef(stop_id, name, d))
                    break
        return out


def locate_along(points: Sequence[XY], cum: List[float], p: XY, after: float = -1.0,
                 tolerance: float = 15.0) -> float:
    """Distance along a polyline of the point closest to p, at or beyond `after`.

    Takes the earliest position within `tolerance` metres of the best match so
    stops on routes that loop back over themselves land on the right pass.
    """
    hits = []
    for i in range(len(points) - 1):
        if cum[i + 1] < after:
            continue
        d, t, _ = project_to_segment(p, points[i], points[i + 1])
        along = cum[i] + t * (cum[i + 1] - cum[i])
        if along < after:
            continue
        hits.append((d, along))
    if not hits:
        return cum[-1] if after > 0 else 0.0
    best = min(h[0] for h in hits)
    return min(a for d, a in hits if d <= best + tolerance)
