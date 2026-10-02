"""The mission: what the pilot decided, saved in the KML and planned again.

What matters is the round trip. A plan exported tonight and imported tomorrow
must come back as the same plan -- fixes, events, leg edits, settings -- with
nothing observed carried along, so tomorrow's weather is the weather it is
solved in.
"""

from __future__ import annotations

import base64
import json
from datetime import date

import pytest

from engine import kml
from engine import mission as ms
from engine import navlog as nl
from engine.geo import LatLon
from server.main import PlanRequest, RouteFileIn, import_route
from server.main import route_kml as export_route

KSQL = {"name": "KSQL", "lat": 37.5119, "lon": -122.2495, "kind": "airport",
        "elevation_ft": 5, "id": "a"}
KSBP = {"name": "KSBP", "lat": 35.2368, "lon": -120.6424, "kind": "airport",
        "elevation_ft": 212, "id": "b"}


def request(**changes) -> PlanRequest:
    body = {
        "waypoints": [
            {**KSQL, "altimeter_inhg": 30.12, "oat_c": 18, "wind_from_deg": 300,
             "wind_speed_kt": 12, "weather_reported": True},
            {**KSBP, "events": [
                {"kind": "complete", "lat": 37.40, "lon": -122.15, "target_altitude_ft": 2500},
                {"kind": "start", "lat": 37.10, "lon": -121.85, "target_altitude_ft": 7500},
            ]},
        ],
        "planning_mode": "hybrid",
        "cruise_altitude_ft": 7500,
        "cruise_rpm": 2300,
        "fuel_on_board_gal": 48,
        "flight_date": "2026-08-15",
        "segment_overrides": [
            {"segment_key": "a>b", "wind_from_deg": 160, "wind_speed_kt": 25},
            {"segment_key": "a>b", "phase": "cruise", "tas_kt": 104},
        ],
        "forecasts": [{"lat": 37.5, "lon": -122.2, "hours": []}],
    }
    body.update(changes)
    return PlanRequest.model_validate(body)


def exported(req: PlanRequest) -> str:
    response = export_route(req)
    assert response.status_code == 200, response.body
    return response.body.decode()


def imported(text: str) -> dict:
    return import_route(RouteFileIn(content_base64=base64.b64encode(text.encode()).decode()))


class TestTheMissionDocument:
    def test_observed_weather_is_left_out(self):
        plan = json.loads(ms.mission_json(request().model_dump(mode="json")))["plan"]
        assert "forecasts" not in plan
        assert "altimeter_inhg" not in plan["waypoints"][0]
        assert "wind_from_deg" not in plan["waypoints"][0]

    def test_the_planners_own_points_are_left_out(self):
        req = request()
        body = req.model_dump(mode="json")
        body["waypoints"].insert(1, {**KSQL, "name": "TOC", "generated": True, "id": None})
        plan = json.loads(ms.mission_json(body))["plan"]
        assert [w["name"] for w in plan["waypoints"]] == ["KSQL", "KSBP"]

    def test_an_unknown_version_is_refused(self):
        with pytest.raises(ms.MissionError, match="version"):
            ms.parse_mission(json.dumps({"schema": 99, "plan": {"waypoints": []}}))


class TestRoundTrip:
    def test_the_plan_comes_back_as_it_was_sent(self):
        req = request()
        result = imported(exported(req))
        mission = result["mission"]
        assert mission["planning_mode"] == "hybrid"
        assert (mission["cruise_altitude_ft"], mission["cruise_rpm"], mission["fuel_on_board_gal"]) == (7500, 2300, 48)
        assert [w["id"] for w in mission["waypoints"]] == ["a", "b"]
        assert mission["waypoints"][1]["events"] == [e.model_dump() for e in req.waypoints[1].events]
        assert mission["segment_overrides"] == [o.model_dump(mode="json") for o in req.segment_overrides]
        assert mission["forecasts"] == []

    def test_it_plans_again_the_same_in_the_same_weather(self):
        """The same plan, once the day's observations are taken off both:
        the field reports are not saved, because they are fetched again."""
        body = request(forecasts=[]).model_dump(mode="json")
        body["waypoints"][0] = {**KSQL, "segment_type": "automatic"}
        req = PlanRequest.model_validate(body)
        mission = imported(exported(req))["mission"]
        from server.main import _build_from_request
        before = _build_from_request(req)[0].navlog
        after = _build_from_request(PlanRequest.model_validate(mission))[0].navlog
        assert [(leg.to_name, round(leg.ete_min, 3)) for leg in after.legs] == [
            (leg.to_name, round(leg.ete_min, 3)) for leg in before.legs]

    def test_the_snapshot_records_where_the_tops_fell(self):
        snapshot = imported(exported(request()))["snapshot"]
        roles = [row["end_role"] for row in snapshot["rows"] if row["end_role"]]
        assert roles[:3] == ["TOC", "BOC", "TOC"]
        assert all(row["segment_key"] == "a>b" for row in snapshot["rows"])
        assert all(row["wind_typed"] for row in snapshot["rows"])

    def test_a_plan_that_does_not_solve_is_still_saved(self):
        req = request(cruise_altitude_ft=100)
        text = exported(req)
        result = imported(text)
        assert result["mission"]["cruise_altitude_ft"] == 100
        assert result["snapshot"] is None

    def test_a_file_without_a_mission_imports_its_route(self):
        log = nl.build_navlog(
            [nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5),
             nl.Waypoint("KSBP", LatLon(35.2368, -120.6424), "airport", elevation_ft=212)],
            7500, conditions=nl.Conditions(flight_date=date(2026, 8, 15)), planning_mode="hybrid")
        result = imported(kml.route_kml(log))
        assert result["mission"] is None
        assert [w["name"] for w in result["waypoints"]] == ["KSQL", "KSBP"]

    def test_a_mission_from_a_newer_version_leaves_the_route(self):
        text = exported(request()).replace('{"schema":1', '{"schema":2')
        result = imported(text)
        assert result["mission"] is None
        assert any("version 2" in w for w in result["warnings"])
        assert [w["name"] for w in result["waypoints"]] == ["KSQL", "KSBP"]

    def test_a_viewer_still_sees_only_the_route(self):
        """The mission is document data: no extra placemarks for Google Earth."""
        text = exported(request())
        names = [w.name for w in kml.parse_kml(text.encode())]
        assert names == ["KSQL", "KSBP"]
