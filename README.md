# gtfs-osm-directions

Generate turn-by-turn driving directions for every path in a GTFS feed
("Head east on West Lane → At the roundabout, take the 2nd exit onto Main
Street → Turn left onto Oak Avenue …") by map-matching the route shapes onto
OpenStreetMap ways.

Pure Python 3.9+, no required dependencies.

## Quick start

```bash
# Try it on the bundled synthetic town
python -m gtfs_osm_directions examples/sample/gtfs --osm examples/sample/town.osm

# Real data: a GTFS zip plus an OSM extract (e.g. from Geofabrik / BBBike)
python -m gtfs_osm_directions feed.zip --osm city.osm -r 10 -f html -o route10.html

# Or let the tool download just the ways it needs from Overpass
python -m gtfs_osm_directions feed.zip --overpass -r 10
```

Or install the `gtfs-osm-directions` command with `pip install .` (add
`.[pbf]` to read `.osm.pbf` files through pyosmium).

Sample output (`examples/sample_output.txt`):

```
Route 10 – West Lane - Station towards Station (direction 0)
  2 trip(s) · shape SH10 · 2.0 km · 100% matched to OSM
------------------------------------------------------------------------
  1. Head east on West Lane  (280 m)
       • stop: West Lane West [S1]
  2. At the roundabout, take the 2nd exit onto Main Street  (550 m)
       • stop: Market Square [S2]
  3. Turn left onto Oak Avenue  (400 m)
       • stop: Oak Avenue Mid [S3]
  4. Turn right onto Elm Road  (400 m)
       • stop: Elm Road East [S4]
  5. Turn right onto Station Road  (390 m)
       • stop: Station [S5]
  6. Arrive at Station
```

## Options

| Option | Meaning |
| --- | --- |
| `--osm FILE` | `.osm`, `.osm.gz`, `.osm.bz2`, or `.osm.pbf` (needs `pip install osmium`) |
| `--overpass` | Download ways for the selected routes' bounding box (cached in `--overpass-cache`) |
| `-r/--route` | `route_id` or `route_short_name`; repeat for several (default: all routes) |
| `--direction 0\|1` | Only one `direction_id` |
| `-f/--format` | `text` (default), `json`, `geojson`, or `html` (Leaflet map with a step list) |
| `-o/--output` | Write to a file instead of stdout |
| `--step` | Shape resample spacing in metres (default 10) |
| `--radius` | Max distance between shape and OSM way in metres (default 35) |
| `--straight-threshold` | Turns smaller than this many degrees are "continue" (default 20) |

## How it works

1. **GTFS → patterns** (`gtfs.py`). Trips are grouped by route, direction and
   `shape_id`, so each distinct path is processed once. The most common stop
   sequence in each group gives the stops and destination.
2. **OSM → network** (`osm.py`, `network.py`). Ways usable by the route's
   mode are kept: roads for buses (honouring `access`, `bus`/`psv`, `oneway`,
   `oneway:bus`, and implied one-way roundabouts/motorways), `railway=tram|light_rail`
   for trams, and heavy-rail/subway tracks for trains. Segments go into a grid
   spatial index in a local metric projection.
3. **Map matching** (`matcher.py`). The shape is resampled every 10 m, and a
   Viterbi (HMM) pass picks one OSM way per sample. The cost combines
   distance, heading agreement and one-way rules. Switching to a connected way
   costs little and jumping to an unconnected way costs a lot, so the match
   doesn't flicker between parallel streets or cross streets.
4. **Directions** (`directions.py`). Consecutive samples on ways with the same
   name/ref become a leg. At each leg change the turn angle is measured 30 m
   either side of the junction:

   | Angle | Instruction |
   | --- | --- |
   | ≤ 20° | Continue straight |
   | ≤ 45° | Bear left/right |
   | ≤ 135° | Turn left/right |
   | ≤ 170° | Sharp left/right |
   | > 170° | U-turn |

   Also handled:
   - Roundabouts: the exit number is counted by walking the ring.
   - Slip roads: `*_link` ways give ramp and merge instructions.
   - Reversals on the same street come out as U-turns.
   - Each stop is attached to the step on which it is served.
5. **No `shapes.txt`?** The path is built by routing between consecutive stops
   on the OSM network (A*, respecting one-way streets), then matched as above.

## Output formats

- `json`: one object per pattern with metadata, `matched_ratio`, warnings and
  instructions (action, modifier, street, text, location, distance, turn angle,
  roundabout exit, stops).
- `geojson`: a LineString per pattern (snapped geometry) plus a Point per
  instruction. Load it in QGIS or geojson.io.
- `html`: a self-contained Leaflet page with a route picker and clickable steps.

## Library use

```python
from gtfs_osm_directions import GTFSFeed, load_osm, DirectionsEngine, feed_projection

feed = GTFSFeed("feed.zip")
patterns = feed.patterns(["10"])
engine = DirectionsEngine(feed, load_osm("city.osm"), feed_projection(feed, patterns))
for p in patterns:
    result = engine.run(p)
    for step in result.instructions:
        print(step.text, round(step.distance))
```

## Limitations

- The OSM extract must cover the whole route. A low `matched_ratio` produces a
  warning.
- Ferries and aerial lifts have no street network and are skipped.
- Tram and rail tracks are often unnamed in OSM, so those directions say
  "unnamed tram" and similar.
- The local equirectangular projection is accurate for city and regional feeds;
  very large (continental) feeds should be split by route.

## Tests

```bash
python -m unittest discover -s tests
```

`tests/sample_data.py` generates the synthetic town used by the tests and
in `examples/sample`.
