"""Load OpenStreetMap ways (.osm / .osm.gz / .osm.bz2 XML, .osm.pbf via pyosmium, or Overpass)."""

from __future__ import annotations

import bz2
import gzip
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@dataclass
class Way:
    way_id: int
    node_ids: List[int]
    tags: Dict[str, str]


@dataclass
class OSMData:
    nodes: Dict[int, Tuple[float, float]]  # id -> (lat, lon)
    ways: Dict[int, Way]


def _is_transport_way(tags: Dict[str, str]) -> bool:
    return "highway" in tags or "railway" in tags


def parse_osm_xml(source) -> OSMData:
    """Parse OSM XML from a path or file object, keeping only highway/railway ways."""
    if isinstance(source, str):
        if source.endswith(".gz"):
            source = gzip.open(source, "rb")
        elif source.endswith(".bz2"):
            source = bz2.open(source, "rb")
    all_nodes: Dict[int, Tuple[float, float]] = {}
    ways: Dict[int, Way] = {}
    for _, elem in ET.iterparse(source, events=("end",)):
        if elem.tag == "node":
            all_nodes[int(elem.get("id"))] = (float(elem.get("lat")), float(elem.get("lon")))
            elem.clear()
        elif elem.tag == "way":
            tags = {t.get("k"): t.get("v") for t in elem.iter("tag")}
            if _is_transport_way(tags):
                nds = [int(n.get("ref")) for n in elem.iter("nd")]
                ways[int(elem.get("id"))] = Way(int(elem.get("id")), nds, tags)
            elem.clear()
        elif elem.tag == "relation":
            elem.clear()
    return _finish(all_nodes, ways)


def parse_osm_pbf(path: str) -> OSMData:
    try:
        import osmium  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Reading .pbf files needs pyosmium (pip install osmium), "
            "or convert to .osm XML first (e.g. osmium cat in.pbf -o out.osm)."
        ) from exc

    ways: Dict[int, Way] = {}
    nodes: Dict[int, Tuple[float, float]] = {}

    class Handler(osmium.SimpleHandler):  # pragma: no cover - optional dependency
        def way(self, w):
            tags = {t.k: t.v for t in w.tags}
            if not _is_transport_way(tags):
                return
            ids = []
            for n in w.nodes:
                ids.append(n.ref)
                if n.location.valid():
                    nodes[n.ref] = (n.location.lat, n.location.lon)
            ways[w.id] = Way(w.id, ids, tags)

    Handler().apply_file(path, locations=True)  # pragma: no cover
    return _finish(nodes, ways)  # pragma: no cover


def _finish(nodes: Dict[int, Tuple[float, float]], ways: Dict[int, Way]) -> OSMData:
    kept_nodes: Dict[int, Tuple[float, float]] = {}
    for way in list(ways.values()):
        way.node_ids = [n for n in way.node_ids if n in nodes]
        if len(way.node_ids) < 2:
            del ways[way.way_id]
            continue
        for n in way.node_ids:
            kept_nodes[n] = nodes[n]
    return OSMData(kept_nodes, ways)


def load_osm(path: str) -> OSMData:
    if path.endswith(".pbf"):
        return parse_osm_pbf(path)
    return parse_osm_xml(path)


def overpass_query(bbox: Tuple[float, float, float, float], modes) -> str:
    """Build an Overpass QL query for the given (south, west, north, east) bbox."""
    s, w, n, e = bbox
    box = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    parts = []
    if "bus" in modes:
        parts.append(f'way["highway"]{box};')
    if "tram" in modes:
        parts.append(f'way["railway"~"^(tram|light_rail)$"]{box};')
    if "rail" in modes:
        parts.append(f'way["railway"~"^(rail|subway|light_rail|narrow_gauge|monorail|funicular)$"]{box};')
    if not parts:
        parts.append(f'way["highway"]{box};')
    body = "".join(parts)
    return f"[out:xml][timeout:180];({body});(._;>;);out body;"


def fetch_overpass(bbox, modes, out_path: str, url: str = OVERPASS_URL) -> str:
    """Download ways for a bbox from Overpass into out_path and return the path."""
    data = urllib.parse.urlencode({"data": overpass_query(bbox, modes)}).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "gtfs-osm-directions/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(out_path, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    return out_path
