"""Hidden-Markov-style map matching of a route shape onto OSM ways.

The shape is resampled at a fixed spacing.  For every sample the candidate
states are the nearby OSM ways (best segment per way).  A Viterbi pass picks
the cheapest sequence, where:

* emission cost grows with distance from the way and with the angle between
  the shape's heading and the way's axis (plus a penalty for driving the
  wrong way down a one-way street), and
* transition cost is free for staying on the same way, cheap for switching to
  a connected way, and expensive for jumping to an unconnected one.

That suppresses flicker at intersections and between parallel streets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .geo import XY, angle_diff, axis_misalignment, bearing_xy
from .network import Network


@dataclass
class MatchParams:
    step: float = 10.0  # resample spacing (m)
    radius: float = 35.0  # candidate search radius (m)
    sigma: float = 10.0  # GPS/shape noise (m)
    heading_weight: float = 4.0
    wrong_way_cost: float = 30.0
    switch_cost: float = 2.0
    jump_cost: float = 12.0
    reverse_cost: float = 15.0


@dataclass
class MatchedPoint:
    xy: XY  # resampled shape point
    heading: float
    way_id: Optional[int] = None
    segment: Optional[int] = None
    forward: bool = True  # travelling along the way's node order
    snapped: Optional[XY] = None


def _headings(points: Sequence[XY]) -> List[float]:
    n = len(points)
    out = []
    for i in range(n):
        a = points[max(0, i - 1)]
        b = points[min(n - 1, i + 1)]
        out.append(bearing_xy(a, b) if a != b else (out[-1] if out else 0.0))
    return out


def match(points: Sequence[XY], net: Network, params: MatchParams = MatchParams()) -> List[MatchedPoint]:
    """Match an already-resampled polyline onto the network."""
    headings = _headings(points)
    result = [MatchedPoint(p, h) for p, h in zip(points, headings)]

    # Candidate states for each sample: way_id -> (cost, segment, forward, snapped)
    cands: List[Dict[int, Tuple[float, int, bool, XY]]] = []
    for p, h in zip(points, headings):
        states: Dict[int, Tuple[float, int, bool, XY]] = {}
        for sid, d, _t, q in net.nearby_segments(p, params.radius):
            seg = net.segments[sid]
            forward = abs(angle_diff(seg.bearing, h)) <= 90.0
            mis = axis_misalignment(seg.bearing, h)
            cost = 0.5 * (d / params.sigma) ** 2 + params.heading_weight * (mis / 90.0) ** 2
            allowed = net.direction[seg.way_id]
            if (allowed == 1 and not forward) or (allowed == -1 and forward):
                cost += params.wrong_way_cost
            prev = states.get(seg.way_id)
            if prev is None or cost < prev[0]:
                states[seg.way_id] = (cost, sid, forward, q)
        cands.append(states)

    # Viterbi, broken into independent chains wherever no candidates exist.
    i, n = 0, len(points)
    while i < n:
        if not cands[i]:
            i += 1
            continue
        j = i
        while j < n and cands[j]:
            j += 1
        _viterbi(cands[i:j], result[i:j], net, params)
        i = j
    return result


def _viterbi(cands, out: List[MatchedPoint], net: Network, params: MatchParams) -> None:
    score: Dict[int, float] = {w: c[0] for w, c in cands[0].items()}
    back: List[Dict[int, int]] = [{}]
    for k in range(1, len(cands)):
        new_score: Dict[int, float] = {}
        ptr: Dict[int, int] = {}
        for w, (cost, _sid, fwd, _q) in cands[k].items():
            best_prev, best_val = None, float("inf")
            for pw, ps in score.items():
                if pw == w:
                    t = 0.0 if cands[k - 1][pw][2] == fwd else params.reverse_cost
                elif net.connected(pw, w):
                    t = params.switch_cost
                else:
                    t = params.jump_cost
                if ps + t < best_val:
                    best_prev, best_val = pw, ps + t
            new_score[w] = best_val + cost
            ptr[w] = best_prev
        score = new_score
        back.append(ptr)

    w = min(score, key=score.get)
    for k in range(len(cands) - 1, -1, -1):
        _cost, sid, fwd, q = cands[k][w]
        mp = out[k]
        mp.way_id, mp.segment, mp.forward, mp.snapped = w, sid, fwd, q
        if k:
            w = back[k][w]
