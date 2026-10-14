"""Render results as text, JSON, GeoJSON or a standalone Leaflet HTML map."""

from __future__ import annotations

import json
from typing import List

from .directions import format_distance
from .pipeline import PatternDirections


def _title(r: PatternDirections) -> str:
    p = r.pattern
    bits = [f"Route {p.route.label}"]
    if p.headsign:
        bits.append(f"towards {p.headsign}")
    if p.direction_id is not None:
        bits.append(f"(direction {p.direction_id})")
    return " ".join(bits)


def to_text(results: List[PatternDirections]) -> str:
    lines: List[str] = []
    for r in results:
        p = r.pattern
        lines.append("=" * 72)
        lines.append(_title(r))
        meta = [f"{len(p.trip_ids)} trip(s)", f"shape {p.shape_id}" if p.shape_id else "no shape (routed between stops)"]
        if r.length_m:
            meta.append(format_distance(r.length_m))
            meta.append(f"{r.matched_ratio:.0%} matched to OSM")
        lines.append("  " + " · ".join(meta))
        for w in r.warnings:
            lines.append(f"  ! {w}")
        lines.append("-" * 72)
        for i, ins in enumerate(r.instructions, 1):
            dist = f"  ({format_distance(ins.distance)})" if ins.distance >= 1 else ""
            lines.append(f"{i:3d}. {ins.text}{dist}")
            for s in ins.stops:
                lines.append(f"       • stop: {s.name} [{s.stop_id}]")
        lines.append("")
    return "\n".join(lines)


def to_json(results: List[PatternDirections]) -> str:
    return json.dumps({"patterns": [r.to_dict() for r in results]}, indent=2, ensure_ascii=False)


def to_geojson_dict(results: List[PatternDirections]) -> dict:
    features = []
    for r in results:
        if not r.geometry:
            continue
        p = r.pattern
        props = {k: v for k, v in r.to_dict().items() if k != "instructions"}
        props["pattern"] = p.key
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[round(lon, 7), round(lat, 7)] for lat, lon in r.geometry]},
            "properties": props,
        })
        for i, ins in enumerate(r.instructions, 1):
            d = ins.to_dict()
            d.pop("location")
            d.update({"pattern": p.key, "route_id": p.route.route_id, "step": i})
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(ins.lon, 7), round(ins.lat, 7)]},
                "properties": d,
            })
    return {"type": "FeatureCollection", "features": features}


def to_geojson(results: List[PatternDirections]) -> str:
    return json.dumps(to_geojson_dict(results), ensure_ascii=False)


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Route Directions</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin: 0; height: 100%; font: 14px/1.4 system-ui, sans-serif; }
  #app { display: flex; height: 100%; }
  #side { width: 380px; overflow-y: auto; border-right: 1px solid #ddd; padding: 12px; box-sizing: border-box; }
  #map { flex: 1; }
  select { width: 100%; padding: 6px; margin-bottom: 8px; }
  ol { padding-left: 22px; margin: 0; }
  li { padding: 6px 0; border-bottom: 1px solid #eee; cursor: pointer; }
  li:hover { background: #f5f7fa; }
  .d { color: #666; font-size: 12px; }
  .s { color: #0a6; font-size: 12px; }
  .w { color: #b40; font-size: 12px; }
  @media (max-width: 700px) { #app { flex-direction: column; } #side { width: 100%; height: 45%; } }
</style>
</head>
<body>
<div id="app"><div id="side"><select id="pick"></select><div id="meta"></div><ol id="steps"></ol></div><div id="map"></div></div>
<script>
const DATA = __DATA__;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(map);
const layer = L.layerGroup().addTo(map);
const pick = document.getElementById('pick');
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = m => m >= 1000 ? (m/1000).toFixed(1) + ' km' : Math.round(m) + ' m';
DATA.forEach((p, i) => { const o = document.createElement('option'); o.value = i; o.textContent = p.title; pick.appendChild(o); });
function show(i) {
  const p = DATA[i];
  layer.clearLayers();
  const line = L.polyline(p.line, {color: '#1565c0', weight: 5, opacity: 0.8}).addTo(layer);
  if (p.line.length) map.fitBounds(line.getBounds(), {padding: [20, 20]});
  document.getElementById('meta').innerHTML = '<div class="d">' + esc(p.meta) + '</div>' +
    p.warnings.map(w => '<div class="w">! ' + esc(w) + '</div>').join('');
  const ol = document.getElementById('steps'); ol.innerHTML = '';
  p.steps.forEach((s, k) => {
    const m = L.circleMarker(s.location, {radius: 6, color: '#fff', weight: 2, fillColor: k === 0 ? '#2e7d32' : (k === p.steps.length - 1 ? '#c62828' : '#ff8f00'), fillOpacity: 1})
      .bindPopup('<b>' + (k + 1) + '.</b> ' + esc(s.text)).addTo(layer);
    const li = document.createElement('li');
    li.innerHTML = esc(s.text) + (s.distance_m >= 1 ? ' <span class="d">(' + fmt(s.distance_m) + ')</span>' : '') +
      s.stops.map(st => '<div class="s">&bull; ' + esc(st.name) + '</div>').join('');
    li.onclick = () => { map.setView(s.location, 17); m.openPopup(); };
    ol.appendChild(li);
  });
}
pick.onchange = () => show(pick.value);
if (DATA.length) show(0); else map.setView([0, 0], 2);
</script>
</body>
</html>
"""


def to_html(results: List[PatternDirections]) -> str:
    data = []
    for r in results:
        d = r.to_dict()
        meta = f"{len(r.pattern.trip_ids)} trip(s) · {format_distance(r.length_m)} · {r.matched_ratio:.0%} matched"
        data.append({
            "title": _title(r),
            "meta": meta,
            "warnings": r.warnings,
            "line": [[round(lat, 6), round(lon, 6)] for lat, lon in r.geometry],
            "steps": d["instructions"],
        })
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__DATA__", payload)


FORMATS = {"text": to_text, "json": to_json, "geojson": to_geojson, "html": to_html}
__all__ = ["FORMATS", "to_text", "to_json", "to_geojson", "to_html"]
