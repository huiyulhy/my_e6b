"""What the API resolves for itself before it plans anything.

Both lookups here are silent when they fail: a waypoint the server does not
recognise as an airport simply arrives at the checklist with no runways and no
pattern altitude, and the checklist reports that as an unknown rather than as
an error. That silence is why they are tested directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from server.main import (
    AloftLevelIn,
    ForecastHourIn,
    WaypointIn,
    _aloft_temperature_samples,
    _pattern_altitude_for,
    _runways_for,
    _winds_aloft,
)

KSQL = {"name": "KSQL", "lat": 37.5119, "lon": -122.2495, "elevation_ft": 5.5}


def waypoint(**overrides):
    return WaypointIn(**{**KSQL, **overrides})


# The database stamps an airport with its size, and that is what a search
# result carries back into the route. Matching only the bare word "airport"
# meant every field added from the search box lost its runways -- the go/no-go
# said "no runway data on file" about fields whose runways it had just listed.
@pytest.mark.parametrize(
    "kind", ["airport", "small_airport", "medium_airport", "large_airport"]
)
def test_every_size_of_airport_is_an_airport(kind):
    assert _runways_for(waypoint(kind=kind))
    assert _pattern_altitude_for(waypoint(kind=kind)) == pytest.approx(800.0)


@pytest.mark.parametrize("kind", ["vfr_waypoint", "town", "vor", "fix"])
def test_nothing_else_is_looked_up_as_one(kind):
    assert _runways_for(waypoint(kind=kind)) == ()
    assert _pattern_altitude_for(waypoint(kind=kind)) is None


def test_a_field_that_publishes_no_pattern_falls_through_to_the_standard_one():
    """None, not a guess: `preflight` is the one place the 1,000 ft lives."""
    livermore = waypoint(name="KLVK", kind="medium_airport")
    assert _pattern_altitude_for(livermore) is None
    assert _runways_for(livermore)


def test_an_identifier_the_database_does_not_have_is_not_an_error():
    assert _runways_for(waypoint(name="ZZZZ", kind="airport")) == ()
    assert _pattern_altitude_for(waypoint(name="ZZZZ", kind="airport")) is None


class TestAloftColumns:
    """What the browser sends back after fetching a column, as the engine sees it.

    The UI fetches `/api/wx/aloft` per leg and hands the levels straight back
    with the plan, so a plan stays one offline call. These are the two
    conversions that happen on the way in.
    """

    def hour(self, **level):
        base = {
            "height_ft": 6000.0,
            "wind_from_deg": 270.0,
            "wind_speed_kt": 25.0,
            "pressure_altitude_ft": 5900.0,
            "isa_deviation_c": 4.0,
        }
        return ForecastHourIn(
            valid_time=datetime(2026, 9, 3, 17, tzinfo=UTC),
            levels=[AloftLevelIn(**{**base, **level})],
        )

    def test_a_level_becomes_a_layer_at_its_geopotential_height(self):
        """Height, not pressure altitude: it is where the aeroplane is."""
        winds = _winds_aloft(self.hour())
        assert winds.layers[0][0] == pytest.approx(6000.0)
        assert winds.at(6000.0).speed_kt == pytest.approx(25.0)

    @pytest.mark.parametrize("missing", ["wind_from_deg", "wind_speed_kt", "height_ft"])
    def test_half_a_wind_is_dropped_rather_than_half_read(self, missing):
        """A direction with no speed would interpolate as a calm from that
        bearing, which is a claim about the air rather than the absence of one."""
        assert _winds_aloft(self.hour(**{missing: None})).layers == ()

    def test_a_temperature_keeps_the_pressure_altitude_it_was_read_at(self):
        """Not the geopotential height: the charts are read at pressure altitude,
        and it is the coordinate every other station's report shares."""
        samples = _aloft_temperature_samples(self.hour())
        assert len(samples) == 1
        assert samples[0].pressure_altitude_ft == pytest.approx(5900.0)
        assert samples[0].isa_deviation_c == pytest.approx(4.0)

    def test_a_level_with_no_temperature_contributes_no_sample(self):
        assert _aloft_temperature_samples(self.hour(isa_deviation_c=None)) == []

    def test_an_empty_hour_is_harmless(self):
        empty = ForecastHourIn(valid_time=datetime(2026, 9, 3, 17, tzinfo=UTC))
        assert _winds_aloft(empty).layers == ()
        assert _aloft_temperature_samples(empty) == []
