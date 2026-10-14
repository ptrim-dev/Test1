"""Small geometry helpers.

Everything is projected into a local equirectangular plane (metres) centred on
the data set.  That is accurate to well under 1% for city/regional-scale
transit networks and keeps the maths simple and dependency free.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

EARTH_RADIUS_M = 6371008.8

XY = Tuple[float, float]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


class LocalProjection:
    """Equirectangular projection around a reference point (x east, y north)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = lat0
        self.lon0 = lon0
        self._ky = math.pi / 180.0 * EARTH_RADIUS_M
        self._kx = self._ky * math.cos(math.radians(lat0))

    @classmethod
    def around(cls, coords: Iterable[Tuple[float, float]]) -> "LocalProjection":
        lats, lons = [], []
        for lat, lon in coords:
            lats.append(lat)
            lons.append(lon)
        if not lats:
            raise ValueError("cannot build a projection from zero coordinates")
        return cls((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)

    def to_xy(self, lat: float, lon: float) -> XY:
        return ((lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky)

    def to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        return (self.lat0 + y / self._ky, self.lon0 + x / self._kx)


def dist(a: XY, b: XY) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def bearing_xy(a: XY, b: XY) -> float:
    """Compass bearing (0 = north, 90 = east) from a to b in the local plane."""
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0


def angle_diff(from_bearing: float, to_bearing: float) -> float:
    """Signed turn angle in (-180, 180]; positive = clockwise (right turn)."""
    d = (to_bearing - from_bearing + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def axis_misalignment(b1: float, b2: float) -> float:
    """Angle in [0, 90] between two undirected axes."""
    d = abs(angle_diff(b1, b2))
    return 180.0 - d if d > 90.0 else d


def project_to_segment(p: XY, a: XY, b: XY) -> Tuple[float, float, XY]:
    """Project p onto segment ab -> (distance, t in [0,1], projected point)."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        t = 0.0
    else:
        t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / l2
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    q = (ax + t * dx, ay + t * dy)
    return math.hypot(p[0] - q[0], p[1] - q[1]), t, q


def cumulative_lengths(points: Sequence[XY]) -> List[float]:
    out = [0.0]
    for i in range(1, len(points)):
        out.append(out[-1] + dist(points[i - 1], points[i]))
    return out


def resample(points: Sequence[XY], step: float) -> List[XY]:
    """Resample a polyline at a fixed spacing, always keeping both ends."""
    pts = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
    if len(pts) < 2:
        return list(pts)
    out = [pts[0]]
    carry = 0.0  # distance travelled since the last emitted point
    for a, b in zip(pts, pts[1:]):
        seg = dist(a, b)
        pos = step - carry
        while pos <= seg:
            t = pos / seg
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            pos += step
        carry = seg - (pos - step)
    if dist(out[-1], pts[-1]) > 1e-6:
        out.append(pts[-1])
    return out


_CARDINALS = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
]


def cardinal(bearing: float) -> str:
    return _CARDINALS[int(((bearing % 360.0) + 22.5) // 45.0) % 8]
