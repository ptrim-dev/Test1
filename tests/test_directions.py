import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gtfs_osm_directions import geo  # noqa: E402
from gtfs_osm_directions.cli import main  # noqa: E402
from gtfs_osm_directions.directions import ordinal  # noqa: E402
from gtfs_osm_directions.gtfs import GTFSFeed  # noqa: E402
from gtfs_osm_directions.network import way_allowed, way_direction  # noqa: E402
from gtfs_osm_directions.osm import load_osm, overpass_query  # noqa: E402
from gtfs_osm_directions.output import to_geojson_dict, to_html, to_text  # noqa: E402
from gtfs_osm_directions.pipeline import DirectionsEngine, feed_projection  # noqa: E402
import sample_data  # noqa: E402


class SampleTownTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.gtfs_dir, cls.osm_path = sample_data.write_sample(cls.tmp.name)
        cls.feed = GTFSFeed(cls.gtfs_dir)
        cls.osm = load_osm(cls.osm_path)
        patterns = cls.feed.patterns()
        engine = DirectionsEngine(cls.feed, cls.osm, feed_projection(cls.feed, patterns))
        cls.results = {p.route.short_name: engine.run(p) for p in patterns}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def actions(self, route):
        return [(i.action, i.modifier, i.street) for i in self.results[route].instructions]

    def test_patterns_group_identical_trips(self):
        r10 = self.results["10"].pattern
        self.assertEqual(r10.trip_ids, ["T10a", "T10b"])
        self.assertEqual(r10.shape_id, "SH10")

    def test_shaped_route_instructions(self):
        self.assertEqual(self.actions("10"), [
            ("depart", None, "West Lane"),
            ("roundabout", None, "Main Street"),
            ("turn", "left", "Oak Avenue"),
            ("turn", "right", "Elm Road"),
            ("turn", "right", "Station Road"),
            ("arrive", None, None),
        ])
        ins = self.results["10"].instructions
        self.assertEqual(ins[1].roundabout_exit, 2)
        self.assertIn("2nd exit onto Main Street", ins[1].text)
        self.assertTrue(ins[0].text.startswith("Head east on West Lane"))
        self.assertEqual(ins[-1].text, "Arrive at Station")
        self.assertEqual(self.results["10"].matched_ratio, 1.0)

    def test_footway_is_ignored_for_buses(self):
        streets = {i.street for i in self.results["10"].instructions}
        self.assertNotIn("Riverside Walk", streets)

    def test_stops_are_attached_to_their_legs(self):
        ins = self.results["10"].instructions
        stops = [[s.stop_id for s in i.stops] for i in ins]
        self.assertEqual(stops, [["S1"], ["S2"], ["S3"], ["S4"], ["S5"], []])
        dists = [s.distance for i in ins for s in i.stops]
        self.assertEqual(dists, sorted(dists))

    def test_distances_sum_to_length(self):
        r = self.results["10"]
        total = sum(i.distance for i in r.instructions)
        self.assertAlmostEqual(total, r.length_m, delta=1.0)
        self.assertGreater(r.length_m, 1900)

    def test_unshaped_route_is_routed_between_stops(self):
        r = self.results["20"]
        self.assertEqual(r.source, "routed-from-stops")
        self.assertEqual(self.actions("20"), [
            ("depart", None, "Main Street"),
            ("uturn", None, "Main Street"),
            ("turn", "left", "Oak Avenue"),
            ("arrive", None, None),
        ])
        self.assertIn("U-turn to stay on Main Street", r.instructions[1].text)

    def test_outputs(self):
        results = list(self.results.values())
        text = to_text(results)
        self.assertIn("At the roundabout, take the 2nd exit onto Main Street", text)
        gj = to_geojson_dict(results)
        kinds = [f["geometry"]["type"] for f in gj["features"]]
        self.assertEqual(kinds.count("LineString"), 2)
        self.assertEqual(kinds.count("Point"), sum(len(r.instructions) for r in results))
        page = to_html(results)
        self.assertIn("leaflet@1.9.4", page)
        self.assertNotIn("__DATA__", page)

    def test_cli_json_and_zip_input(self):
        zpath = os.path.join(self.tmp.name, "feed.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for name in os.listdir(self.gtfs_dir):
                zf.write(os.path.join(self.gtfs_dir, name), name)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main([zpath, "--osm", self.osm_path, "-r", "10", "-f", "json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(len(data["patterns"]), 1)
        self.assertEqual(data["patterns"][0]["instructions"][1]["roundabout_exit"], 2)

    def test_cli_unknown_route(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main([self.gtfs_dir, "--osm", self.osm_path, "-r", "nope"]), 1)


class UnitTest(unittest.TestCase):
    def test_resample_keeps_ends_and_spacing(self):
        pts = geo.resample([(0, 0), (25, 0), (25, 25)], 10)
        self.assertEqual(pts[0], (0, 0))
        self.assertEqual(pts[-1], (25, 25))
        for a, b in zip(pts[:-2], pts[1:-1]):
            self.assertLessEqual(geo.dist(a, b), 10.0001)

    def test_angles(self):
        self.assertEqual(geo.angle_diff(0, 90), 90)
        self.assertEqual(geo.angle_diff(350, 10), 20)
        self.assertEqual(geo.angle_diff(10, 350), -20)
        self.assertEqual(geo.cardinal(44), "northeast")
        self.assertEqual(geo.cardinal(359), "north")

    def test_projection_round_trip(self):
        p = geo.LocalProjection(40.0, -74.0)
        lat, lon = p.to_latlon(*p.to_xy(40.01, -73.99))
        self.assertAlmostEqual(lat, 40.01, places=9)
        self.assertAlmostEqual(lon, -73.99, places=9)

    def test_ordinals(self):
        self.assertEqual([ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)],
                         ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"])

    def test_way_rules(self):
        self.assertTrue(way_allowed({"highway": "residential"}, "bus"))
        self.assertFalse(way_allowed({"highway": "footway"}, "bus"))
        self.assertTrue(way_allowed({"highway": "pedestrian", "bus": "yes"}, "bus"))
        self.assertFalse(way_allowed({"highway": "service", "access": "private"}, "bus"))
        self.assertTrue(way_allowed({"railway": "tram"}, "tram"))
        self.assertFalse(way_allowed({"highway": "primary"}, "rail"))
        self.assertEqual(way_direction({"oneway": "yes"}, "bus"), 1)
        self.assertEqual(way_direction({"oneway": "yes", "oneway:bus": "no"}, "bus"), 0)
        self.assertEqual(way_direction({"junction": "roundabout"}, "bus"), 1)
        self.assertEqual(way_direction({"oneway": "-1"}, "bus"), -1)

    def test_overpass_query(self):
        q = overpass_query((1, 2, 3, 4), {"bus", "rail"})
        self.assertIn('way["highway"](1.000000,2.000000,3.000000,4.000000);', q)
        self.assertIn("railway", q)
        self.assertTrue(q.endswith("out body;"))


if __name__ == "__main__":
    unittest.main()
