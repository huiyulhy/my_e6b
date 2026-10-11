"""The pre-dispatch sheet: takeoff and landing at every field it names.

Departure and destination both ways round, plus an alternate the route never
visits. Each is read at that field's own weather, and none of the extra
operations may move the go/no-go verdict -- they are information beside it.
"""

from __future__ import annotations

import json

import pytest

from engine import mission as ms
from engine import navlog as nl
from engine.geo import LatLon
from server.main import PlanRequest, plan

KSQL = {"name": "KSQL", "lat": 37.5119, "lon": -122.2495, "kind": "airport",
        "elevation_ft": 5.5, "id": "a"}
KMRY = {"name": "KMRY", "lat": 36.5870, "lon": -121.8428, "kind": "airport",
        "elevation_ft": 256.6, "id": "b"}
KHAF = {"name": "KHAF", "lat": 37.5134, "lon": -122.5012, "kind": "airport",
        "elevation_ft": 66.3}


def request(**changes) -> PlanRequest:
    body = {
        "waypoints": [
            {**KSQL, "altimeter_inhg": 30.02, "oat_c": 20},
            {**KMRY, "segment_type": "cruise", "altimeter_inhg": 30.00, "oat_c": 16},
        ],
        "cruise_altitude_ft": 4500,
        "flight_date": "2026-08-15",
    }
    body.update(changes)
    return PlanRequest.model_validate(body)


def by_role(response: dict) -> dict[str, dict]:
    return {field["role"]: field for field in response["dispatch"]}


class TestDepartureAndDestination:
    def test_both_operations_at_both_ends(self):
        response = plan(request())
        assert response["ok"], response.get("error")
        fields = by_role(response)
        assert set(fields) == {"departure", "destination"}
        for field in fields.values():
            assert field["ok"]
            for op in ("takeoff", "landing"):
                runway = field[op]["runways"][0]
                assert runway["ground_roll_ft"] > 0
                assert runway["over_50ft_ft"] > runway["ground_roll_ft"]

    def test_the_same_numbers_the_go_no_go_reads(self):
        """One function computes both, so they cannot disagree."""
        response = plan(request())
        checklist = {
            (c["airport"], c["operation"]): c for c in response["checklist"]["airports"]
        }
        fields = by_role(response)
        assert fields["departure"]["takeoff"] == checklist[("KSQL", "takeoff")]
        assert fields["destination"]["landing"] == checklist[("KMRY", "landing")]

    def test_each_field_is_read_at_its_own_setting(self):
        fields = by_role(plan(request()))
        assert fields["departure"]["altimeter_inhg"] == pytest.approx(30.02)
        assert fields["destination"]["altimeter_inhg"] == pytest.approx(30.00)


class TestAlternate:
    def test_read_at_its_own_weather(self):
        calm = by_role(plan(request(alternate={**KHAF, "oat_c": 15})))["alternate"]
        windy = by_role(plan(request(alternate={
            **KHAF, "oat_c": 15, "wind_from_deg": 300, "wind_speed_kt": 15,
        })))["alternate"]
        assert calm["ok"] and windy["ok"]
        assert calm["landing"]["runways"][0]["headwind_kt"] is None
        assert windy["landing"]["runways"][0]["headwind_kt"] > 0
        assert (
            windy["landing"]["runways"][0]["over_50ft_ft"]
            < calm["landing"]["runways"][0]["over_50ft_ft"]
        )

    def test_never_part_of_the_verdict(self):
        """An alternate under an obscuration is a poor alternate, not a no-go."""
        clear = plan(request())
        fogged = plan(request(alternate={
            **KHAF, "visibility_sm": 0.25, "ceiling_ft_agl": 100,
            "ceiling_cover": "VV", "sky_reported": True, "weather_reported": True,
        }))
        assert by_role(fogged)["alternate"]["landing"]["passes"] is False
        assert fogged["checklist"]["verdict"] == clear["checklist"]["verdict"]
        assert fogged["checklist"]["blockers"] == clear["checklist"]["blockers"]

    def test_a_field_with_no_elevation_is_refused_on_its_own(self):
        response = plan(request(alternate={**KHAF, "elevation_ft": None}))
        assert response["ok"]
        fields = by_role(response)
        assert fields["alternate"]["ok"] is False
        assert fields["departure"]["ok"] and fields["destination"]["ok"]


def test_field_checks_needs_an_elevation():
    waypoint = nl.Waypoint(name="FIX", position=LatLon(37.0, -122.0))
    assert nl.field_checks(waypoint, nl.Aircraft(), nl.Conditions()) is None


def test_the_mission_keeps_the_alternate_but_not_its_weather():
    req = request(alternate={**KHAF, "altimeter_inhg": 29.95, "wind_speed_kt": 8})
    saved = json.loads(ms.mission_json(req.model_dump(mode="json")))["plan"]
    assert saved["alternate"]["name"] == "KHAF"
    assert "altimeter_inhg" not in saved["alternate"]
    assert "wind_speed_kt" not in saved["alternate"]
