"""Winds and temperatures aloft, off an Open-Meteo pressure-level payload.

The payloads here are synthetic but shaped exactly like the real thing --
naive GMT hour strings, one flat list per `<field>_<level>hPa`, a top-level
`elevation` in metres. What is asserted is the part that is easy to get
quietly wrong: which altitude each product is keyed by, and what happens to a
level the model invented underneath a mountain.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from engine import aloft
from engine.atmosphere import isa_temperature_c

HOURS = [
    "2026-09-01T14:00",
    "2026-09-01T15:00",
    "2026-09-01T16:00",
]
NOON = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)

# Three levels with round numbers, so every expected value can be read off by
# hand: a westerly backing with height, and air 10 C warmer than standard.
LEVELS = {
    1000: {"height_m": 100.0, "dir": 270.0, "speed": 10.0, "temp_offset": 10.0},
    850: {"height_m": 1500.0, "dir": 300.0, "speed": 30.0, "temp_offset": 10.0},
    700: {"height_m": 3000.0, "dir": 320.0, "speed": 50.0, "temp_offset": 10.0},
}


def payload(*, elevation_m: float = 5.0, levels=None, drop_fields=()):
    """An Open-Meteo forecast object for the three levels above."""
    levels = LEVELS if levels is None else levels
    hourly: dict = {"time": list(HOURS)}
    for hpa, spec in levels.items():
        pa_ft = aloft.LevelSample(hpa, 0.0).pressure_altitude_ft
        oat = isa_temperature_c(pa_ft) + spec["temp_offset"]
        for name, value in (
            (f"geopotential_height_{hpa}hPa", spec["height_m"]),
            (f"wind_direction_{hpa}hPa", spec["dir"]),
            (f"wind_speed_{hpa}hPa", spec["speed"]),
            (f"temperature_{hpa}hPa", oat),
        ):
            if name in drop_fields:
                continue
            # The middle hour is the one every test targets; the others carry
            # obviously wrong values so a mis-picked slot fails loudly.
            hourly[name] = [value + 100.0, value, value + 100.0]
    return {"elevation": elevation_m, "hourly": hourly}


def parse(**kwargs):
    return aloft.parse_aloft(
        kwargs.pop("payload", None) or payload(**kwargs.pop("payload_kwargs", {})),
        kwargs.pop("target", NOON),
        levels=kwargs.pop("levels", (1000, 850, 700)),
        **kwargs,
    )


# --- which levels to ask for ---------------------------------------------


class TestLevelSelection:
    def test_one_level_above_the_ceiling_is_kept(self):
        # 850 hPa is 4781 ft, 800 hPa is 6394 ft. A flight to 5500 needs both
        # or the top of the climb is held flat from below.
        chosen = aloft.levels_up_to(5500.0)
        assert chosen[-1] == 800
        assert 850 in chosen

    def test_a_ceiling_on_a_level_does_not_need_the_one_above_it(self):
        assert aloft.levels_up_to(4781.0)[-1] == 850

    def test_a_sea_level_ceiling_still_brackets_something(self):
        assert aloft.levels_up_to(0.0) == (1000,)

    def test_the_field_names_are_the_ones_open_meteo_publishes(self):
        assert aloft.hourly_fields((850,)) == (
            "temperature_850hPa",
            "wind_speed_850hPa",
            "wind_direction_850hPa",
            "geopotential_height_850hPa",
        )


# --- the two keys --------------------------------------------------------


class TestPressureAltitudeIsExact:
    """A constant-pressure surface has a pressure altitude by definition."""

    @pytest.mark.parametrize(
        "hpa, expected_ft", [(1000, 364.0), (850, 4781.0), (700, 9882.0)]
    )
    def test_a_level_knows_its_own_pressure_altitude(self, hpa, expected_ft):
        assert aloft.LevelSample(hpa, 0.0).pressure_altitude_ft == pytest.approx(
            expected_ft, abs=1.0
        )

    def test_and_it_does_not_depend_on_the_height_the_model_reports(self):
        # Same level, a 2000 ft different geometric height -- a real day-to-day
        # variation. The pressure altitude cannot move, and that is the whole
        # reason the temperature profile is keyed by it.
        low = aloft.LevelSample(850, 4000.0).pressure_altitude_ft
        high = aloft.LevelSample(850, 6000.0).pressure_altitude_ft
        assert low == high


class TestKeys:
    def test_wind_is_keyed_by_geopotential_height(self):
        forecast = parse()
        layers = forecast.winds().layers
        heights = [round(altitude) for altitude, _ in layers]
        # 100 m, 1500 m, 3000 m in feet -- not the pressure altitudes.
        assert heights == [328, 4921, 9843]

    def test_temperature_is_keyed_by_pressure_altitude(self):
        profile = parse().temperatures()
        assert [round(s.pressure_altitude_ft) for s in profile.samples] == [
            364,
            4781,
            9882,
        ]

    def test_the_two_are_different_altitudes_for_the_same_level(self):
        level = parse().levels[1]
        assert level.height_ft == pytest.approx(4921.3, abs=1.0)
        assert level.pressure_altitude_ft == pytest.approx(4781.0, abs=1.0)


# --- what comes out ------------------------------------------------------


class TestProducts:
    def test_a_level_reads_back_its_own_wind(self):
        assert parse().wind_at(4921.3).speed_kt == pytest.approx(30.0, abs=0.1)

    def test_between_levels_the_wind_is_blended_as_a_vector(self):
        forecast = parse()
        midway = forecast.wind_at((4921.3 + 9843.0) / 2)
        # Between 300/30 and 320/50: the direction lands between the two and
        # the speed between the two, which a scalar average of degrees would
        # also do here -- the point is that it is not either endpoint.
        assert 300.0 < midway.from_deg < 320.0
        assert 30.0 < midway.speed_kt < 50.0

    def test_below_the_lowest_level_the_bottom_wind_is_held(self):
        assert parse().wind_at(0.0).speed_kt == pytest.approx(10.0)

    def test_the_temperature_profile_reproduces_each_level(self):
        forecast = parse()
        for level in forecast.levels:
            assert forecast.oat_at(level.pressure_altitude_ft) == pytest.approx(
                level.oat_c, abs=0.01
            )

    def test_a_uniform_isa_offset_comes_back_as_that_deviation(self):
        profile = parse().temperatures()
        assert [round(s.isa_deviation_c, 1) for s in profile.samples] == [10.0] * 3

    def test_deviation_is_held_flat_above_the_top_level(self):
        # Not the temperature: holding a temperature would claim the air stops
        # cooling with height. This is `TemperatureProfile`'s rule, asserted
        # here because the whole point of feeding it deviations is to get it.
        profile = parse().temperatures()
        assert profile.deviation_at(20000.0) == pytest.approx(10.0, abs=0.1)
        assert profile.oat_at_pressure_altitude(20000.0) < profile.samples[-1].oat_c


# --- what gets thrown away -----------------------------------------------


class TestLevelsBelowGround:
    """A model reports 1000 hPa over a 6000 ft mountain by inventing air."""

    def high_terrain(self):
        return parse(payload=payload(elevation_m=1800.0))  # ~5900 ft, Truckee

    def test_a_level_under_the_model_terrain_is_dropped(self):
        forecast = self.high_terrain()
        assert [level.pressure_hpa for level in forecast.levels] == [700.0]

    def test_and_the_forecast_says_which_and_why(self):
        note = " ".join(self.high_terrain().notes)
        assert "1000 hPa" in note and "850 hPa" in note
        assert "below the model's ground" in note

    def test_the_terrain_it_measured_against_is_reported(self):
        assert self.high_terrain().terrain_elevation_ft == pytest.approx(
            5905.5, abs=1.0
        )

    def test_a_payload_with_no_elevation_drops_nothing(self):
        # Without a terrain height there is no basis to call a level
        # underground, and guessing one would throw away real data.
        bare = payload()
        del bare["elevation"]
        forecast = parse(payload=bare)
        assert len(forecast.levels) == 3
        assert forecast.terrain_elevation_ft is None


class TestMissingPieces:
    def test_a_level_with_no_height_cannot_be_placed_and_says_so(self):
        forecast = parse(
            payload=payload(drop_fields=("geopotential_height_850hPa",))
        )
        assert [level.pressure_hpa for level in forecast.levels] == [1000.0, 700.0]
        assert "could not be placed" in " ".join(forecast.notes)

    def test_half_a_wind_is_no_wind_but_keeps_the_temperature(self):
        forecast = parse(payload=payload(drop_fields=("wind_speed_850hPa",)))
        level = next(lvl for lvl in forecast.levels if lvl.pressure_hpa == 850)
        assert level.wind is None
        assert level.oat_c is not None
        # And the profile simply spans the gap rather than inventing a calm.
        assert len(forecast.winds().layers) == 2
        assert forecast.has_temperature

    def test_a_forecast_with_no_temperatures_falls_back_to_the_typed_deviation(self):
        stripped = payload(
            drop_fields=tuple(f"temperature_{hpa}hPa" for hpa in (1000, 850, 700))
        )
        forecast = parse(payload=stripped)
        assert not forecast.has_temperature
        assert forecast.oat_at(5000.0) is None
        assert forecast.temperatures(default_deviation_c=7.0).deviation_at(
            5000.0
        ) == pytest.approx(7.0)

    def test_a_forecast_with_no_winds_reads_as_calm(self):
        stripped = payload(
            drop_fields=tuple(
                name
                for hpa in (1000, 850, 700)
                for name in (f"wind_speed_{hpa}hPa", f"wind_direction_{hpa}hPa")
            )
        )
        forecast = parse(payload=stripped)
        assert not forecast.has_wind
        assert forecast.wind_at(6000.0).speed_kt == 0.0


class TestTimeAndShape:
    def test_the_hour_nearest_the_target_is_taken(self):
        forecast = parse(target=datetime(2026, 9, 1, 15, 20, tzinfo=UTC))
        assert forecast.valid_time == datetime(2026, 9, 1, 15, 0, tzinfo=UTC)
        # The neighbouring hours carry +100 kt; picking one would be obvious.
        assert forecast.wind_at(4921.3).speed_kt == pytest.approx(30.0, abs=0.1)

    def test_a_target_outside_the_window_is_refused_rather_than_snapped(self):
        assert parse(target=NOON + timedelta(days=2)) is None

    def test_a_naive_target_is_read_as_utc(self):
        assert parse(target=NOON.replace(tzinfo=None)) is not None

    def test_a_batched_response_is_indexed(self):
        both = [payload(elevation_m=1800.0), payload(elevation_m=5.0)]
        assert len(parse(payload=both, index=1).levels) == 3
        assert len(parse(payload=both, index=0).levels) == 1
        assert parse(payload=both, index=9) is None

    @pytest.mark.parametrize(
        "junk", [None, {}, [], "no", {"hourly": {}}, {"hourly": {"time": []}}]
    )
    def test_an_unusable_payload_is_none_rather_than_an_empty_forecast(self, junk):
        # None is the absence of a claim. An empty forecast would read as
        # "calm and standard", which is a claim about the day.
        assert aloft.parse_aloft(junk, NOON) is None


class TestFeedingItToAPlan:
    """The seam: samples go in, not a finished profile.

    `build_navlog` builds one temperature curve for the whole route out of
    everything it can find. Handing it a profile would see it discarded and
    rebuilt from the fields alone -- which is exactly the bug this test
    exists to keep out.
    """

    def test_samples_are_already_in_pressure_altitude(self):
        samples = parse().temperature_samples()
        assert [round(s.pressure_altitude_ft) for s in samples] == [364, 4781, 9882]
        assert all(s.isa_deviation_c == pytest.approx(10.0, abs=0.1) for s in samples)

    def test_a_level_with_no_temperature_contributes_no_sample(self):
        forecast = parse(payload=payload(drop_fields=("temperature_850hPa",)))
        assert len(forecast.temperature_samples()) == 2

    def test_the_plan_reads_the_forecast_temperature_at_cruise(self):
        from datetime import date

        from engine import navlog as nl
        from engine import preflight as pf
        from engine.geo import LatLon

        forecast = parse()
        departure = nl.Waypoint(
            "KSQL", LatLon(37.512, -122.250), kind="airport", elevation_ft=5.0,
            runways=(pf.Runway("12/30", 2600.0, "ASPH"),),
        )
        destination = nl.Waypoint(
            "KMRY", LatLon(36.587, -121.843), kind="airport", elevation_ft=257.0,
            runways=(pf.Runway("10R/28L", 7616.0, "ASPH"),),
        )
        aircraft = nl.Aircraft(
            weight_lb=2400.0, cruise_rpm=2400.0, fuel_on_board_gal=40.0
        )
        log = nl.build_navlog(
            [departure, destination],
            6500.0,
            aircraft,
            nl.Conditions(
                flight_date=date(2026, 9, 1),
                winds=forecast.winds(),
                temperatures_aloft=forecast.temperature_samples(),
            ),
            planning_mode="auto",
        )
        cruise = next(leg for leg in log.legs if leg.phase == "cruise")
        # The whole column is ISA+10, and no field reported anything to pull
        # the curve away from it, so cruise is standard plus ten.
        assert cruise.oat_c == pytest.approx(
            isa_temperature_c(cruise.pressure_altitude_ft) + 10.0, abs=0.5
        )
        # And the wind at cruise is the forecast's, not calm.
        assert cruise.wind_speed_kt > 0.0
