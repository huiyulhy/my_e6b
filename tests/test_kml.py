"""KML export and import tests.

The file has two readers: a viewer, which wants every point at its planned
altitude, and this program, which wants the pilot's route back exactly. What
is tested is that both get what they need from the same document, and that
somebody else's file is read for what it has rather than refused for what it
lacks.
"""

import io
import xml.etree.ElementTree as ET
import zipfile
from datetime import date

import pytest

from engine import kml
from engine import navlog as nl
from engine.atmosphere import M_PER_FT
from engine.geo import LatLon

KSQL = nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5)
BRIDGE = nl.Waypoint("Bridge", LatLon(36.8, -121.5), "waypoint", altitude_ft=4500)
KSBP = nl.Waypoint("KSBP", LatLon(35.2368, -120.6424), "airport", elevation_ft=212)

CALM = nl.Conditions(flight_date=date(2026, 8, 15))


def log(route=(KSQL, BRIDGE, KSBP), **kwargs):
    return nl.build_navlog(list(route), 7500, conditions=CALM, planning_mode="auto", **kwargs)


def local(tag):
    return tag.rsplit("}", 1)[-1]


def placemarks(text):
    root = ET.fromstring(text)
    return [n for n in root.iter() if local(n.tag) == "Placemark" and n.find("{*}Point") is not None]


def coordinate(placemark):
    lon, lat, alt = placemark.find("{*}Point/{*}coordinates").text.split(",")
    return float(lon), float(lat), float(alt)


def extended(placemark):
    return {
        d.get("name"): (d.find("{*}value").text or "")
        for d in placemark.find("{*}ExtendedData")
    }


class TestExport:
    def test_one_placemark_per_resolved_waypoint_including_toc_and_tod(self):
        navlog = log()
        names = [p.find("{*}name").text for p in placemarks(kml.route_kml(navlog))]
        assert names == [w.name for w in navlog.resolved_waypoints]
        assert "TOC" in names and "TOD" in names

    def test_the_coordinate_is_the_planned_altitude_in_metres(self):
        navlog = log()
        by_name = {p.find("{*}name").text: p for p in placemarks(kml.route_kml(navlog))}
        cruise = next(leg for leg in navlog.legs if leg.phase == "cruise")
        _lon, _lat, toc_alt = coordinate(by_name["TOC"])
        assert toc_alt == pytest.approx(cruise.entry_altitude_ft * M_PER_FT, abs=0.1)
        # The departure is at the field, not at cruise.
        _lon, _lat, ksql_alt = coordinate(by_name["KSQL"])
        assert ksql_alt == pytest.approx(5 * M_PER_FT, abs=0.1)
        # And the destination at its pattern altitude, which is where the
        # descent ends: 212 ft + 1,000, to the nearest hundred.
        _lon, _lat, ksbp_alt = coordinate(by_name["KSBP"])
        assert ksbp_alt == pytest.approx(1200 * M_PER_FT, abs=0.5)

    def test_altitudes_are_absolute(self):
        text = kml.route_kml(log())
        for p in placemarks(text):
            assert p.find("{*}Point/{*}altitudeMode").text == "absolute"

    def test_extended_data_carries_the_constraint_not_the_planned_altitude(self):
        by_name = {p.find("{*}name").text: p for p in placemarks(kml.route_kml(log()))}
        # KSQL has no constraint and its planned altitude is the field's.
        assert extended(by_name["KSQL"])["e6b:altitude_ft"] == ""
        assert extended(by_name["KSQL"])["e6b:kind"] == "airport"
        assert extended(by_name["KSQL"])["e6b:generated"] == "0"
        assert extended(by_name["Bridge"])["e6b:altitude_ft"] == "4500"
        assert extended(by_name["TOC"])["e6b:generated"] == "1"

    def test_the_route_is_also_a_line_through_the_points(self):
        root = ET.fromstring(kml.route_kml(log()))
        line = next(n for n in root.iter() if local(n.tag) == "LineString")
        vertices = line.find("{*}coordinates").text.split()
        assert len(vertices) == len(log().resolved_waypoints)

    def test_a_name_that_is_markup_is_escaped(self):
        odd = nl.Waypoint("<b>&", LatLon(36.8, -121.5), "waypoint")
        text = kml.route_kml(log((KSQL, odd, KSBP)))
        assert "<b>&" not in text
        assert any(p.find("{*}name").text == "<b>&" for p in placemarks(text))

    def test_the_filename_names_the_route(self):
        assert kml.kml_filename(log()) == "route-KSQL-KSBP.kml"


class TestRoundTrip:
    def test_our_own_file_comes_back_as_the_pilots_route(self):
        navlog = log()
        points = kml.parse_kml(kml.route_kml(navlog).encode())
        assert [p.name for p in points] == ["KSQL", "Bridge", "KSBP"]
        assert all(p.from_extended_data for p in points)

    def test_the_planners_points_are_dropped(self):
        names = [p.name for p in kml.parse_kml(kml.route_kml(log()).encode())]
        assert "TOC" not in names and "TOD" not in names

    def test_constraints_and_kinds_survive(self):
        by_name = {p.name: p for p in kml.parse_kml(kml.route_kml(log()).encode())}
        assert by_name["Bridge"].altitude_ft == 4500
        assert by_name["Bridge"].kind == "waypoint"
        assert by_name["KSQL"].altitude_ft is None
        assert by_name["KSQL"].kind == "airport"
        assert by_name["KSQL"].elevation_ft == 5
        assert by_name["KSBP"].segment_type == "descent"

    def test_a_landing_stop_survives(self):
        stop = nl.Waypoint(
            "KMRY", LatLon(36.5870, -121.8429), "airport", elevation_ft=257, is_landing=True
        )
        by_name = {
            p.name: p for p in kml.parse_kml(kml.route_kml(log((KSQL, stop, KSBP))).encode())
        }
        assert by_name["KMRY"].is_landing is True
        assert by_name["KSQL"].is_landing is False

    def test_a_kmz_reads_the_same(self):
        text = kml.route_kml(log())
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("doc.kml", text)
        assert kml.parse_kml(buffer.getvalue()) == kml.parse_kml(text.encode())


GOOGLE_EARTH = b"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
<Document><name>My places</name>
  <Placemark><name>ksql</name><Point><coordinates>-122.2495,37.5119,0</coordinates></Point></Placemark>
  <Placemark><name>Hill</name><Point><altitudeMode>absolute</altitudeMode>
    <coordinates>-121.5,36.8,1524</coordinates></Point></Placemark>
  <Placemark><Point><coordinates>-120.6424,35.2368</coordinates></Point></Placemark>
</Document></kml>"""

PATH_ONLY = b"""<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Path</name><LineString><tessellate>1</tessellate>
    <coordinates>
      -122.2495,37.5119,0
      -121.5,36.8,0
      -120.6424,35.2368,0
    </coordinates></LineString></Placemark>
</Document></kml>"""


class TestSomebodyElsesFile:
    def test_placemarks_without_our_data_are_plain_waypoints(self):
        points = kml.parse_kml(GOOGLE_EARTH)
        assert [p.name for p in points] == ["ksql", "Hill", "WP1"]
        assert all(not p.from_extended_data for p in points)
        assert all(p.kind == "waypoint" for p in points)
        assert all(p.segment_type == "automatic" for p in points)

    def test_a_coordinate_altitude_is_read_as_feet(self):
        hill = kml.parse_kml(GOOGLE_EARTH)[1]
        assert hill.altitude_ft == 5000

    def test_zero_altitude_means_no_altitude(self):
        """Zero in KML is "on the ground", not "at sea level"."""
        points = kml.parse_kml(GOOGLE_EARTH)
        assert points[0].altitude_ft is None
        assert points[2].altitude_ft is None

    def test_a_path_alone_is_a_waypoint_per_vertex(self):
        points = kml.parse_kml(PATH_ONLY)
        assert [p.name for p in points] == ["WP1", "WP2", "WP3"]
        assert points[1].lat == 36.8 and points[1].lon == -121.5

    def test_a_namespace_is_not_required(self):
        bare = b"<kml><Placemark><name>A</name><Point><coordinates>-121,36</coordinates></Point></Placemark></kml>"
        assert kml.parse_kml(bare)[0].name == "A"


class TestRefusals:
    def test_a_document_type_declaration_is_refused(self):
        hostile = b'<?xml version="1.0"?><!DOCTYPE kml [<!ENTITY a "aaaa">]><kml>&a;</kml>'
        with pytest.raises(kml.KmlError, match="document type"):
            kml.parse_kml(hostile)

    def test_an_oversize_file_is_refused(self):
        big = b"<kml>" + b" " * (kml.MAX_BYTES + 1) + b"</kml>"
        with pytest.raises(kml.KmlError, match="larger than"):
            kml.parse_kml(big)

    def test_an_oversize_kmz_member_is_refused_before_it_is_inflated(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("doc.kml", b"<kml>" + b" " * (kml.MAX_BYTES + 1) + b"</kml>")
        with pytest.raises(kml.KmlError, match="larger than"):
            kml.parse_kml(buffer.getvalue())

    def test_a_polar_point_is_refused(self):
        polar = b"<kml><Placemark><name>N</name><Point><coordinates>0,89</coordinates></Point></Placemark></kml>"
        with pytest.raises(kml.KmlError, match="latitude"):
            kml.parse_kml(polar)

    def test_an_empty_document_is_refused(self):
        with pytest.raises(kml.KmlError, match="no placemarks"):
            kml.parse_kml(b"<kml><Document/></kml>")

    def test_malformed_xml_is_refused(self):
        with pytest.raises(kml.KmlError, match="well-formed"):
            kml.parse_kml(b"<kml><Placemark>")

    def test_something_that_is_not_a_zip_is_refused(self):
        with pytest.raises(kml.KmlError, match="KMZ"):
            kml.parse_kml(b"PK\x03\x04 not really")
