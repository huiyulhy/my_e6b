"""Garmin flight plan (.fpl) export and import tests.

The export's readers are ForeFlight and Garmin devices, which refuse a whole
file over one character the schema does not allow. What is tested is that the
file keeps to the schema's patterns whatever the pilot named things, that the
pilot's names still come through, and that airports go in as airports. The
import is tested on our own files and on files shaped as other programs
write them.
"""

import base64
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import pytest

from engine import fpl
from engine import navlog as nl
from engine.geo import LatLon
from server.main import PlanRequest, RouteFileIn, import_route, route_fpl

NS = {"fp": fpl.FPL_NS}

KSQL = nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5)
BRIDGE = nl.Waypoint("Golden Gate Bridge", LatLon(37.8199, -122.4783), "waypoint")
KSBP = nl.Waypoint("KSBP", LatLon(35.2368, -120.6424), "airport", elevation_ft=212)
TOC = nl.Waypoint("TOC", LatLon(37.6, -122.3), "phase", generated=True)

IDENTIFIER = re.compile(r"^[A-Z0-9]{1,12}$")
TEXT = re.compile(r"^([A-Z0-9 /]{1,25})?$")


def parse(text):
    return ET.fromstring(text)


def table(root):
    return [
        {local(c.tag): (c.text or "") for c in w}
        for w in root.findall("fp:waypoint-table/fp:waypoint", NS)
    ]


def route_points(root):
    return [
        (
            p.findtext("fp:waypoint-identifier", namespaces=NS),
            p.findtext("fp:waypoint-type", namespaces=NS),
        )
        for p in root.findall("fp:route/fp:route-point", NS)
    ]


def local(tag):
    return tag.rsplit("}", 1)[-1]


class TestDocument:
    def test_root_is_a_flight_plan_in_garmins_namespace(self):
        root = parse(fpl.flight_plan([KSQL, BRIDGE, KSBP]))
        assert root.tag == f"{{{fpl.FPL_NS}}}flight-plan"

    def test_elements_come_in_schema_order(self):
        root = parse(fpl.flight_plan([KSQL, BRIDGE, KSBP]))
        assert [local(c.tag) for c in root] == ["created", "waypoint-table", "route"]
        entry = root.find("fp:waypoint-table/fp:waypoint", NS)
        assert [local(c.tag) for c in entry] == [
            "identifier",
            "type",
            "country-code",
            "lat",
            "lon",
            "comment",
        ]
        route = root.find("fp:route", NS)
        assert [local(c.tag) for c in route][:2] == ["route-name", "flight-plan-index"]

    def test_created_is_utc_with_a_z(self):
        when = datetime(2026, 10, 10, 15, 30, tzinfo=UTC)
        root = parse(fpl.flight_plan([KSQL, KSBP], created=when))
        assert root.findtext("fp:created", namespaces=NS) == "2026-10-10T15:30:00Z"

    def test_route_name_is_departure_to_destination(self):
        root = parse(fpl.flight_plan([KSQL, BRIDGE, KSBP]))
        assert root.findtext("fp:route/fp:route-name", namespaces=NS) == "KSQL TO KSBP"


class TestWaypoints:
    def test_airports_go_in_by_identifier_and_the_rest_as_user_waypoints(self):
        root = parse(fpl.flight_plan([KSQL, BRIDGE, KSBP]))
        assert route_points(root) == [
            ("KSQL", "AIRPORT"),
            ("GOLDENGATEBR", "USER WAYPOINT"),
            ("KSBP", "AIRPORT"),
        ]

    def test_the_pilots_label_is_the_comment(self):
        root = parse(fpl.flight_plan([KSQL, BRIDGE, KSBP]))
        by_id = {w["identifier"]: w for w in table(root)}
        assert by_id["GOLDENGATEBR"]["comment"] == "GOLDEN GATE BRIDGE"

    def test_a_user_waypoint_keeps_its_position(self):
        root = parse(fpl.flight_plan([KSQL, BRIDGE, KSBP]))
        entry = next(w for w in table(root) if w["type"] == "USER WAYPOINT")
        assert float(entry["lat"]) == pytest.approx(37.8199)
        assert float(entry["lon"]) == pytest.approx(-122.4783)
        assert entry["country-code"] == ""

    def test_planner_inserted_points_are_left_out(self):
        root = parse(fpl.flight_plan([KSQL, TOC, KSBP]))
        assert [i for i, _ in route_points(root)] == ["KSQL", "KSBP"]

    def test_every_field_keeps_to_the_schema_whatever_the_name(self):
        odd = [
            nl.Waypoint("lake—élan #3 (north shore, by the dam)", LatLon(37.0, -122.0)),
            nl.Waypoint("!!!", LatLon(36.9, -121.9)),
            nl.Waypoint("Señor's & Co.", LatLon(36.8, -121.8), "vfr_waypoint"),
        ]
        root = parse(fpl.flight_plan([KSQL, *odd, KSBP]))
        for entry in table(root):
            assert IDENTIFIER.match(entry["identifier"]), entry
            assert TEXT.match(entry["comment"]), entry
        assert TEXT.match(root.findtext("fp:route/fp:route-name", namespaces=NS))

    def test_a_name_with_nothing_usable_is_numbered_by_its_place(self):
        blank = nl.Waypoint("???", LatLon(36.9, -121.9))
        root = parse(fpl.flight_plan([KSQL, blank, KSBP]))
        assert route_points(root)[1] == ("WP2", "USER WAYPOINT")

    def test_two_points_that_fold_to_one_identifier_each_get_their_own(self):
        a = nl.Waypoint("Reservoir", LatLon(37.0, -122.0))
        b = nl.Waypoint("reservoir!", LatLon(36.5, -121.5))
        root = parse(fpl.flight_plan([KSQL, a, b, KSBP]))
        assert [i for i, _ in route_points(root)][1:3] == ["RESERVOIR", "RESERVOIR2"]
        assert len(table(root)) == 4

    def test_a_point_visited_twice_is_one_table_entry_named_twice(self):
        here = nl.Waypoint("Bridge", LatLon(37.0, -122.0))
        root = parse(fpl.flight_plan([KSQL, here, KSBP, here, KSQL]))
        assert [i for i, _ in route_points(root)] == ["KSQL", "BRIDGE", "KSBP", "BRIDGE", "KSQL"]
        assert len(table(root)) == 3

    def test_a_user_waypoint_named_like_an_airport_does_not_take_its_identifier(self):
        fake = nl.Waypoint("KSQL", LatLon(37.0, -122.0), "waypoint")
        root = parse(fpl.flight_plan([KSQL, fake, KSBP]))
        assert route_points(root)[1] == ("KSQL2", "USER WAYPOINT")

    def test_an_airport_whose_name_is_no_identifier_goes_in_at_its_position(self):
        renamed = nl.Waypoint("Home field", LatLon(37.5119, -122.2495), "airport")
        root = parse(fpl.flight_plan([renamed, KSBP]))
        assert route_points(root)[0] == ("HOMEFIELD", "USER WAYPOINT")


class TestRefusals:
    def test_a_single_point_is_not_a_flight_plan(self):
        with pytest.raises(fpl.FplError):
            fpl.flight_plan([KSQL, TOC])


class TestFilename:
    def test_named_by_the_ends_of_the_route(self):
        assert fpl.fpl_filename([KSQL, BRIDGE, TOC, KSBP]) == "route-KSQL-KSBP.fpl"


class TestEndpoint:
    def request(self, *points):
        return PlanRequest(
            waypoints=[
                {
                    "name": w.name,
                    "lat": w.position.lat,
                    "lon": w.position.lon,
                    "kind": w.kind,
                    "generated": w.generated,
                }
                for w in points
            ]
        )

    def test_returns_the_file_as_an_attachment(self):
        response = route_fpl(self.request(KSQL, BRIDGE, TOC, KSBP))
        assert response.status_code == 200
        assert 'filename="route-KSQL-KSBP.fpl"' in response.headers["content-disposition"]
        root = parse(response.body)
        assert [i for i, _ in route_points(root)] == ["KSQL", "GOLDENGATEBR", "KSBP"]

    def test_too_short_a_route_is_a_400(self):
        response = route_fpl(self.request(KSQL))
        assert response.status_code == 400


# A flight plan as ForeFlight writes one: Garmin's namespace, airports with
# their ICAO region, a user waypoint with no comment.
FOREFLIGHT = """<?xml version="1.0" encoding="utf-8"?>
<flight-plan xmlns="http://www8.garmin.com/xmlschemas/FlightPlan/v1">
  <created>2026-10-10T15:30:00Z</created>
  <waypoint-table>
    <waypoint>
      <identifier>KSQL</identifier>
      <type>AIRPORT</type>
      <country-code>K2</country-code>
      <lat>37.511853</lat>
      <lon>-122.249522</lon>
      <comment />
    </waypoint>
    <waypoint>
      <identifier>KSBP</identifier>
      <type>AIRPORT</type>
      <country-code>K2</country-code>
      <lat>35.236806</lat>
      <lon>-120.642389</lon>
      <comment />
    </waypoint>
    <waypoint>
      <identifier>LAKE</identifier>
      <type>USER WAYPOINT</type>
      <country-code />
      <lat>37.0</lat>
      <lon>-122.0</lon>
      <comment>CREATED 10-OCT-26</comment>
      <elevation>304.8</elevation>
    </waypoint>
  </waypoint-table>
  <route>
    <route-name>KSQL TO KSBP</route-name>
    <flight-plan-index>1</flight-plan-index>
    <route-point>
      <waypoint-identifier>KSQL</waypoint-identifier>
      <waypoint-type>AIRPORT</waypoint-type>
      <waypoint-country-code>K2</waypoint-country-code>
    </route-point>
    <route-point>
      <waypoint-identifier>LAKE</waypoint-identifier>
      <waypoint-type>USER WAYPOINT</waypoint-type>
      <waypoint-country-code />
    </route-point>
    <route-point>
      <waypoint-identifier>KSBP</waypoint-identifier>
      <waypoint-type>AIRPORT</waypoint-type>
      <waypoint-country-code>K2</waypoint-country-code>
    </route-point>
  </route>
</flight-plan>
"""


class TestImport:
    def test_our_own_export_comes_back_with_its_labels(self):
        reservoir = nl.Waypoint("Reservoir", LatLon(37.0, -122.0))
        read = fpl.read_fpl(fpl.flight_plan([KSQL, BRIDGE, reservoir, TOC, KSBP]).encode())
        assert [p.name for p in read.waypoints] == [
            "KSQL",
            "GOLDEN GATE BRIDGE",
            "RESERVOIR",
            "KSBP",
        ]
        assert [p.kind for p in read.waypoints] == ["airport", "waypoint", "waypoint", "airport"]
        assert read.waypoints[1].lat == pytest.approx(37.8199)
        assert read.warnings == []

    def test_a_route_is_read_in_flying_order_not_table_order(self):
        read = fpl.read_fpl(FOREFLIGHT.encode())
        assert [p.name for p in read.waypoints] == ["KSQL", "LAKE", "KSBP"]

    def test_a_comment_the_identifier_was_not_made_from_is_not_the_name(self):
        lake = fpl.read_fpl(FOREFLIGHT.encode()).waypoints[1]
        assert lake.name == "LAKE"

    def test_elevation_is_read_as_metres(self):
        lake = fpl.read_fpl(FOREFLIGHT.encode()).waypoints[1]
        assert lake.elevation_ft == 1000

    def test_a_point_visited_twice_comes_back_twice(self):
        here = nl.Waypoint("Bridge", LatLon(37.0, -122.0))
        read = fpl.read_fpl(fpl.flight_plan([KSQL, here, KSBP, here, KSQL]).encode())
        assert [p.name for p in read.waypoints] == ["KSQL", "BRIDGE", "KSBP", "BRIDGE", "KSQL"]

    def test_a_file_without_the_namespace_is_read_alike(self):
        bare = FOREFLIGHT.replace(' xmlns="http://www8.garmin.com/xmlschemas/FlightPlan/v1"', "")
        read = fpl.read_fpl(bare.encode())
        assert [p.name for p in read.waypoints] == ["KSQL", "LAKE", "KSBP"]

    def test_a_file_with_no_route_is_its_table_with_a_warning(self):
        no_route = re.sub(r"<route>.*</route>", "", FOREFLIGHT, flags=re.S)
        read = fpl.read_fpl(no_route.encode())
        assert [p.name for p in read.waypoints] == ["KSQL", "KSBP", "LAKE"]
        assert read.warnings

    def test_a_route_point_missing_from_the_table_is_left_out_with_a_warning(self):
        broken = FOREFLIGHT.replace(
            "<waypoint-identifier>LAKE</waypoint-identifier>",
            "<waypoint-identifier>POND</waypoint-identifier>",
        )
        read = fpl.read_fpl(broken.encode())
        assert [p.name for p in read.waypoints] == ["KSQL", "KSBP"]
        assert any("POND" in w for w in read.warnings)


# What ForeFlight actually writes: declared UTF-8 but saved as UTF-16 with a
# byte-order mark, no country codes or comments, its own altitude per point,
# an empty type on a lat/lon point.
FOREFLIGHT_REAL = """<?xml version="1.0" encoding="utf-8"?>
<flight-plan xmlns="http://www8.garmin.com/xmlschemas/FlightPlan/v1">
<created>20261015T14:18:12Z</created>
<aircraft><aircraft-tailnumber>N494SP</aircraft-tailnumber></aircraft>
<flight-data><etd-zulu></etd-zulu><altitude-ft>5500</altitude-ft></flight-data>
<waypoint-table>
    <waypoint>
        <identifier>KPAO</identifier><type>AIRPORT</type>
        <lat>37.461121</lat><lon>-122.115047</lon><altitude-ft></altitude-ft>
    </waypoint>
    <waypoint>
        <identifier>VPBDX</identifier><type>INT</type>
        <lat>37.491622</lat><lon>-122.139181</lon><altitude-ft>3500</altitude-ft>
    </waypoint>
    <waypoint>
        <identifier>38.33337N/121.60583W</identifier><type></type>
        <lat>38.33336701562786</lat><lon>-121.6058368759383</lon><altitude-ft></altitude-ft>
    </waypoint>
    <waypoint>
        <identifier>KSMF</identifier><type>AIRPORT</type>
        <lat>38.695444</lat><lon>-121.590778</lon><altitude-ft></altitude-ft>
    </waypoint>
</waypoint-table>
<route>
    <route-name>KPAO TO KSMF</route-name>
    <route-point><waypoint-identifier>KPAO</waypoint-identifier><waypoint-type>AIRPORT</waypoint-type></route-point>
    <route-point><waypoint-identifier>VPBDX</waypoint-identifier><waypoint-type>INT</waypoint-type></route-point>
    <route-point><waypoint-identifier>38.33337N/121.60583W</waypoint-identifier><waypoint-type></waypoint-type></route-point>
    <route-point><waypoint-identifier>KSMF</waypoint-identifier><waypoint-type>AIRPORT</waypoint-type></route-point>
</route>
</flight-plan>
"""


class TestForeFlightFile:
    def test_utf16_with_a_mark_declared_as_utf8_is_read(self):
        data = b"\xff\xfe" + FOREFLIGHT_REAL.encode("utf-16-le")
        assert fpl.looks_like_fpl(data)
        read = fpl.read_fpl(data)
        assert [p.name for p in read.waypoints] == ["KPAO", "VPBDX", "38.33337N/121.60583W", "KSMF"]

    def test_utf8_with_a_mark_is_read(self):
        read = fpl.read_fpl(b"\xef\xbb\xbf" + FOREFLIGHT_REAL.encode())
        assert read.waypoints[0].name == "KPAO"

    def test_missing_country_code_and_comment_are_not_a_problem(self):
        read = fpl.read_fpl(FOREFLIGHT_REAL.encode())
        assert [p.kind for p in read.waypoints] == ["airport", "waypoint", "waypoint", "airport"]
        assert read.warnings == []

    def test_foreflights_altitude_per_point_is_the_crossing_altitude(self):
        read = fpl.read_fpl(FOREFLIGHT_REAL.encode())
        assert [p.altitude_ft for p in read.waypoints] == [None, 3500, None, None]

    def test_a_genuinely_undecodable_file_is_refused_not_crashed(self):
        with pytest.raises(fpl.FplError, match="not readable"):
            fpl.read_fpl(b"\xff\xfe" + b"\x00\xd8")  # a lone surrogate


class TestImportRefusals:
    def test_a_doctype_is_refused(self):
        evil = FOREFLIGHT.replace(
            '<?xml version="1.0" encoding="utf-8"?>',
            '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]>',
        )
        with pytest.raises(fpl.FplError, match="document type"):
            fpl.read_fpl(evil.encode())

    def test_a_position_that_is_not_a_number_is_refused(self):
        with pytest.raises(fpl.FplError, match="LAKE"):
            fpl.read_fpl(FOREFLIGHT.replace("<lat>37.0</lat>", "<lat>north</lat>").encode())

    def test_a_position_beyond_the_model_is_refused(self):
        with pytest.raises(fpl.FplError, match="latitude"):
            fpl.read_fpl(FOREFLIGHT.replace("<lat>37.0</lat>", "<lat>89.0</lat>").encode())

    def test_an_empty_flight_plan_is_refused(self):
        empty = '<flight-plan xmlns="http://www8.garmin.com/xmlschemas/FlightPlan/v1"/>'
        with pytest.raises(fpl.FplError):
            fpl.read_fpl(empty.encode())


class TestSniffing:
    def test_a_flight_plan_is_recognised_by_its_root(self):
        assert fpl.looks_like_fpl(FOREFLIGHT.encode())
        assert fpl.looks_like_fpl(b"<fp:flight-plan xmlns:fp='x'/>")

    def test_kml_is_not_a_flight_plan(self):
        kml = b'<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document/></kml>'
        assert not fpl.looks_like_fpl(kml)


class TestImportEndpoint:
    def upload(self, text):
        return import_route(RouteFileIn(content_base64=base64.b64encode(text.encode()).decode()))

    def test_airports_are_resolved_and_user_waypoints_kept_where_they_are(self):
        result = self.upload(FOREFLIGHT)
        assert result["ok"]
        ksql, lake, ksbp = result["waypoints"]
        assert ksql["name"] == "KSQL" and ksql["elevation_ft"] is not None
        assert ksbp["name"] == "KSBP"
        assert lake == {
            "name": "LAKE",
            "lat": 37.0,
            "lon": -122.0,
            "kind": "waypoint",
            "elevation_ft": 1000,
            "altitude_ft": None,
            "segment_type": "automatic",
            "is_landing": False,
            "label": "",
        }
        assert result["mission"] is None

    def test_a_user_waypoint_named_like_an_airport_is_not_looked_up(self):
        result = self.upload(FOREFLIGHT.replace(">LAKE<", ">O88<"))
        assert result["waypoints"][1]["kind"] == "waypoint"

    def test_an_airport_the_database_does_not_have_is_kept_with_a_warning(self):
        result = self.upload(FOREFLIGHT.replace(">KSBP<", ">ZZZZ<"))
        assert result["waypoints"][2]["kind"] == "waypoint"
        assert any("ZZZZ" in w for w in result["warnings"])

    def test_a_broken_flight_plan_is_a_400(self):
        response = self.upload(FOREFLIGHT.replace("<lat>37.0</lat>", "<lat>north</lat>"))
        assert response.status_code == 400
