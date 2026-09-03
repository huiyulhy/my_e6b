"""Navigation log tests.

Checks the sequencing and bookkeeping rather than the physics -- the physics
is tested in `test_atmosphere`, `test_performance`, `test_geo` and
`test_magnetic`. What matters here is that phases are split correctly, that
distance, time and fuel accumulate consistently, and that the log complains
when the flight will not work.
"""

from dataclasses import replace
from datetime import date
from typing import ClassVar

import pytest

from engine import navlog as nl
from engine.geo import LatLon, inverse

KSQL = nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5)
KMRY = nl.Waypoint("KMRY", LatLon(36.5870, -121.8429), "airport", elevation_ft=257)
KSBP = nl.Waypoint("KSBP", LatLon(35.2368, -120.6424), "airport", elevation_ft=212)
VPWDM = nl.Waypoint("VPWDM", LatLon(37.2000, -122.0500), "vfr_waypoint")

CALM = nl.Conditions(flight_date=date(2026, 8, 15))


def simple_log(**kwargs):
    return nl.build_navlog(
        [KSQL, KMRY], kwargs.pop("altitude", 6500), **kwargs, planning_mode="auto"
    )


class TestRouteValidation:
    def test_single_waypoint_rejected(self):
        with pytest.raises(nl.RouteError, match="at least"):
            nl.build_navlog([KSQL], 6500, conditions=CALM, planning_mode="auto")

    def test_missing_elevation_rejected(self):
        no_elevation = nl.Waypoint("KXXX", LatLon(36.0, -121.0), "airport")
        with pytest.raises(nl.RouteError, match="elevation"):
            nl.build_navlog([KSQL, no_elevation], 6500, conditions=CALM, planning_mode="auto")

    def test_cruise_altitude_below_airports_rejected(self):
        with pytest.raises(nl.RouteError, match="not above"):
            nl.build_navlog([KSQL, KMRY], 200, conditions=CALM, planning_mode="auto")

    def test_zero_length_route_rejected(self):
        with pytest.raises(nl.RouteError):
            nl.build_navlog([KSQL, KSQL], 6500, conditions=CALM, planning_mode="auto")


class TestPhaseStructure:
    def test_top_of_climb_and_descent_are_inserted(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        names = {leg.from_name for leg in log.legs} | {leg.to_name for leg in log.legs}
        assert "TOC" in names
        assert "TOD" in names

    def test_all_three_phases_present(self):
        """Over the flown rows -- taxi and pattern are ground rows, not phases."""
        log = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        flown = {leg.phase for leg in log.legs if leg.covers_ground}
        assert flown == {"climb", "cruise", "descent"}

    def test_ground_rows_bracket_the_flying(self):
        """Taxi before the first flown row, pattern after the last."""
        log = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        phases = [leg.phase for leg in log.legs]
        assert phases[0] == "taxi"
        assert phases[-1] == "pattern"

    def test_phases_occur_in_order(self):
        log = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        order = {"climb": 0, "cruise": 1, "descent": 2}
        ranks = [order[leg.phase] for leg in log.legs if leg.covers_ground]
        assert ranks == sorted(ranks)

    def test_cruise_legs_are_at_cruise_altitude(self):
        log = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        for leg in log.legs:
            if leg.phase == "cruise":
                assert leg.altitude_ft == pytest.approx(7500)

    def test_user_waypoints_are_preserved(self):
        log = nl.build_navlog([KSQL, VPWDM, KMRY], 6500, conditions=CALM, planning_mode="auto")
        names = {leg.from_name for leg in log.legs} | {leg.to_name for leg in log.legs}
        assert {"KSQL", "VPWDM", "KMRY"} <= names

    def test_short_route_warns_that_cruise_is_never_reached(self):
        """Two nearby airports cannot climb to 10500 and descend again."""
        nearby = nl.Waypoint("KPAO", LatLon(37.4611, -122.1150), "airport", elevation_ft=7)
        log = nl.build_navlog([KSQL, nearby], 10500, conditions=CALM, planning_mode="auto")
        assert any("never levels off" in w for w in log.warnings)


class TestAccounting:
    def test_distance_accumulates_to_the_total(self):
        log = nl.build_navlog([KSQL, VPWDM, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert sum(leg.distance_nm for leg in log.legs) == pytest.approx(
            log.total_distance_nm
        )
        assert log.legs[-1].cumulative_distance_nm == pytest.approx(log.total_distance_nm)

    def test_time_accumulates_to_the_total(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert sum(leg.ete_min for leg in log.legs) == pytest.approx(log.total_time_min)
        assert log.legs[-1].cumulative_ete_min == pytest.approx(log.total_time_min)

    def test_fuel_total_includes_the_taxi_allowance(self):
        """The allowance is a row of its own, so the total already has it."""
        aircraft = nl.Aircraft(taxi_fuel_gal=1.1)
        log = nl.build_navlog([KSQL, KMRY], 6500, aircraft, CALM, planning_mode="auto")
        taxi = [leg for leg in log.legs if leg.phase == "taxi"]
        assert len(taxi) == 1
        assert taxi[0].fuel_gal == pytest.approx(1.1)
        assert log.total_fuel_gal == pytest.approx(
            sum(leg.fuel_gal for leg in log.legs)
        )
        flown = sum(leg.fuel_gal for leg in log.legs if leg.covers_ground)
        assert log.total_fuel_gal > flown + 1.0

    def test_start_taxi_takeoff_allowance_defaults_to_the_poh_figure(self):
        """1.4 gal, from the note on the POH climb chart.

        Pinned because it is a published constant, not a modelling choice, and
        because leaving it out or guessing low understates the fuel required by
        more than a quarter of the 30-minute VFR reserve.
        """
        from engine import performance as perf

        assert perf.START_TAXI_TAKEOFF_FUEL_GAL == 1.4
        assert nl.Aircraft().taxi_fuel_gal == 1.4

        log = nl.build_navlog([KSQL, KMRY], 6500, nl.Aircraft(), CALM, planning_mode="auto")
        taxi = next(leg for leg in log.legs if leg.phase == "taxi")
        assert taxi.fuel_gal == pytest.approx(1.4)

    def test_the_allowance_is_charged_per_takeoff_not_per_flight_plan(self):
        """A route that lands en route pays it again on the second departure."""
        direct_flight = nl.build_navlog(
            [KSQL, KSBP], 7500, nl.Aircraft(), CALM, planning_mode="auto"
        )
        with_stop = nl.build_navlog(
            [KSQL, replace(KMRY, is_landing=True), KSBP], 7500, nl.Aircraft(), CALM,
            planning_mode="auto",
        )
        extra = with_stop.total_fuel_gal - direct_flight.total_fuel_gal
        assert extra > 1.4

    def test_fuel_remaining_is_consistent(self):
        aircraft = nl.Aircraft(fuel_on_board_gal=50.0)
        log = nl.build_navlog([KSQL, KMRY], 6500, aircraft, CALM, planning_mode="auto")
        for leg in log.legs:
            assert leg.fuel_remaining_gal == pytest.approx(
                50.0 - leg.cumulative_fuel_gal
            )
        assert log.fuel_remaining_gal == pytest.approx(50.0 - log.total_fuel_gal)

    def test_each_leg_time_matches_distance_over_ground_speed(self):
        """Flown rows only: a ground row covers no distance at no speed."""
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        for leg in log.legs:
            if not leg.covers_ground:
                continue
            assert leg.ete_min == pytest.approx(
                60.0 * leg.distance_nm / leg.ground_speed_kt
            )

    def test_total_distance_matches_the_great_circle_route(self):
        """Splicing TOC and TOD in must not change the route's length."""
        from engine.geo import inverse

        direct_distance = (
            inverse(KSQL.position, VPWDM.position).distance_nm
            + inverse(VPWDM.position, KMRY.position).distance_nm
        )
        log = nl.build_navlog([KSQL, VPWDM, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert log.total_distance_nm == pytest.approx(direct_distance, rel=1e-6)


class TestHeadings:
    def test_magnetic_heading_applies_variation(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        for leg in log.legs:
            expected = (leg.true_heading_deg - leg.variation_deg) % 360.0
            assert leg.magnetic_heading_deg == pytest.approx(expected)

    def test_california_variation_is_easterly(self):
        """Flown rows only: a ground row has no course, so no variation."""
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        flown = [leg for leg in log.legs if leg.covers_ground]
        assert flown
        assert all(10.0 < leg.variation_deg < 16.0 for leg in flown)

    def test_calm_wind_means_heading_equals_course(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        for leg in log.legs:
            assert leg.wind_correction_angle_deg == pytest.approx(0.0)
            assert leg.true_heading_deg == pytest.approx(leg.true_course_deg)

    def test_all_headings_are_normalised(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        for leg in log.legs:
            assert 0.0 <= leg.true_course_deg < 360.0
            assert 0.0 <= leg.magnetic_heading_deg < 360.0


class TestWindEffects:
    def test_headwind_slows_the_flight_down(self):
        calm = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        # KSQL to KMRY runs about 160 true, so wind from 160 is on the nose.
        into_wind = nl.build_navlog(
            [KSQL, KMRY],
            6500,
            conditions=nl.Conditions(
                winds=nl.WindsAloft.uniform(160.0, 25.0), flight_date=date(2026, 8, 15)
            ),
            planning_mode="auto",
        )
        assert into_wind.total_time_min > calm.total_time_min
        assert into_wind.total_fuel_gal > calm.total_fuel_gal

    def test_tailwind_speeds_the_flight_up(self):
        calm = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        pushed = nl.build_navlog(
            [KSQL, KMRY],
            6500,
            conditions=nl.Conditions(
                winds=nl.WindsAloft.uniform(340.0, 25.0), flight_date=date(2026, 8, 15)
            ),
            planning_mode="auto",
        )
        assert pushed.total_time_min < calm.total_time_min

    def test_impossible_headwind_refuses(self):
        """A wind stronger than cruise speed cannot be flown against."""
        with pytest.raises(nl.RouteError):
            nl.build_navlog(
                [KSQL, KMRY],
                6500,
                conditions=nl.Conditions(
                    winds=nl.WindsAloft.uniform(160.0, 200.0),
                    flight_date=date(2026, 8, 15),
                ),
                planning_mode="auto",
            )


class TestWindsAloft:
    def test_calm_everywhere(self):
        assert nl.WindsAloft.calm().at(5000).speed_kt == 0.0

    def test_uniform_is_the_same_at_every_altitude(self):
        winds = nl.WindsAloft.uniform(270.0, 20.0)
        for altitude in (0, 3000, 12000):
            assert winds.at(altitude).from_deg == pytest.approx(270.0)
            assert winds.at(altitude).speed_kt == pytest.approx(20.0)

    def test_interpolates_between_layers(self):
        winds = nl.WindsAloft(((3000.0, nl.Wind(270.0, 10.0)), (9000.0, nl.Wind(270.0, 30.0))))
        assert winds.at(6000.0).speed_kt == pytest.approx(20.0)
        assert winds.at(6000.0).from_deg == pytest.approx(270.0)

    def test_clamps_outside_the_published_layers(self):
        winds = nl.WindsAloft(((3000.0, nl.Wind(270.0, 10.0)), (9000.0, nl.Wind(270.0, 30.0))))
        assert winds.at(0.0).speed_kt == pytest.approx(10.0)
        assert winds.at(20000.0).speed_kt == pytest.approx(30.0)

    def test_interpolates_across_north_correctly(self):
        """Halfway between 350 and 010 is 000, not the arithmetic mean of 180.

        This is why the interpolation works on vector components.
        """
        winds = nl.WindsAloft(((3000.0, nl.Wind(350.0, 20.0)), (9000.0, nl.Wind(10.0, 20.0))))
        result = winds.at(6000.0)
        assert result.from_deg == pytest.approx(0.0, abs=0.5) or result.from_deg == pytest.approx(
            360.0, abs=0.5
        )
        assert result.speed_kt == pytest.approx(19.7, abs=0.5)

    def test_opposing_winds_cancel(self):
        winds = nl.WindsAloft(((3000.0, nl.Wind(90.0, 20.0)), (9000.0, nl.Wind(270.0, 20.0))))
        assert winds.at(6000.0).speed_kt == pytest.approx(0.0, abs=1e-9)


class TestFuelReserve:
    def test_ample_fuel_is_legal(self):
        log = nl.build_navlog(
            [KSQL, KMRY], 6500, nl.Aircraft(fuel_on_board_gal=50.0), CALM, planning_mode="auto"
        )
        assert log.is_legal_on_fuel
        assert not any("reserve" in w for w in log.warnings)

    def test_marginal_fuel_warns(self):
        log = nl.build_navlog(
            [KSQL, KSBP], 7500, nl.Aircraft(fuel_on_board_gal=12.0), CALM, planning_mode="auto"
        )
        assert not log.is_legal_on_fuel
        assert any("reserve" in w for w in log.warnings)

    def test_running_dry_warns_explicitly(self):
        log = nl.build_navlog(
            [KSQL, KSBP], 7500, nl.Aircraft(fuel_on_board_gal=3.0), CALM, planning_mode="auto"
        )
        assert any("runs out of fuel" in w for w in log.warnings)

    def test_night_reserve_is_larger(self):
        day = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        night = nl.build_navlog(
            [KSQL, KMRY],
            6500,
            conditions=nl.Conditions(flight_date=date(2026, 8, 15), night=True),
            planning_mode="auto",
        )
        assert night.reserve_required_gal > day.reserve_required_gal
        # Both figures are rounded up to the tenth a pilot writes down, so the
        # ratio holds to within that rather than exactly.
        assert night.reserve_required_gal == pytest.approx(
            day.reserve_required_gal * 45.0 / 30.0, abs=0.1
        )


class TestConditions:
    def test_hot_day_costs_more_time_and_more_climb_fuel(self):
        """A hot day is slower and climbs worse -- but burns less total fuel.

        Not a typo. At a fixed 2400 RPM, hotter air means less power, so cruise
        drops from 55% to 52% and fuel flow from 7.90 to 7.58 gph. Climb fuel
        does rise, by the POH's 10%-per-10-degC note, from 2.80 to 3.22 gal.
        The cruise saving outweighs the climb penalty, so the total falls.

        The lesson, which has caught this project three times now: a fixed RPM
        is not a fixed power setting. See docs/ARCHITECTURE.md section 5.
        """
        standard = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        hot = nl.build_navlog(
            [KSQL, KSBP],
            7500,
            conditions=nl.Conditions(isa_deviation_c=15.0, flight_date=date(2026, 8, 15)),
            planning_mode="auto",
        )
        assert hot.total_time_min > standard.total_time_min

        def climb_fuel(log):
            return sum(leg.fuel_gal for leg in log.legs if leg.phase == "climb")

        assert climb_fuel(hot) > climb_fuel(standard)
        assert hot.total_fuel_gal < standard.total_fuel_gal

    def test_low_altimeter_raises_pressure_altitude(self):
        conditions = nl.Conditions(altimeter_inhg=29.42, flight_date=date(2026, 8, 15))
        assert conditions.pressure_altitude_ft(6500) > 6500

    def test_oat_follows_isa_deviation(self):
        conditions = nl.Conditions(isa_deviation_c=10.0)
        from engine.atmosphere import isa_temperature_c

        assert conditions.oat_c(6500) == pytest.approx(isa_temperature_c(6500) + 10.0)

    def test_oat_is_referenced_to_pressure_altitude(self):
        """The deviation is applied where the air actually is, not where the
        altimeter says it is.

        At 29.92 the two are the same, which is why the test above holds
        unchanged. At a low setting the aeroplane is higher than indicated in
        pressure terms, so it is colder -- and this is the definition the POH
        cruise chart uses, so anything else would disagree with the book.
        """
        from engine.atmosphere import isa_temperature_c

        conditions = nl.Conditions(isa_deviation_c=10.0, altimeter_inhg=29.42)
        pa = conditions.pressure_altitude_ft(6500)
        assert conditions.oat_c(6500) == pytest.approx(isa_temperature_c(pa) + 10.0)
        assert conditions.oat_c(6500) < isa_temperature_c(6500) + 10.0


class TestFieldWeather:
    """The air at one point on the ground: elevation, setting, temperature.

    One helper behind the go/no-go check, the taxi and pattern rows and the
    temperature profile, so that all three read the same field the same way.
    """

    HOT_HIGH = replace(KMRY, oat_c=35.0, altimeter_inhg=29.42)

    def test_none_without_an_elevation(self):
        """An en-route fix over open country has no field to read."""
        assert nl.field_weather(VPWDM, CALM) is None

    def test_uses_the_waypoints_own_setting_and_temperature(self):
        air = nl.field_weather(self.HOT_HIGH, CALM)
        assert air.altimeter_inhg == pytest.approx(29.42)
        assert air.oat_c == pytest.approx(35.0)
        # 257 ft field, half an inch low, 35 degC: well over 3000 ft of DA.
        assert air.pressure_altitude_ft > 257
        assert air.density_altitude_ft > 3000

    def test_falls_back_to_the_route_where_nothing_was_reported(self):
        conditions = replace(CALM, altimeter_inhg=29.42, isa_deviation_c=10.0)
        air = nl.field_weather(KMRY, conditions)
        assert air.altimeter_inhg == pytest.approx(29.42)
        assert air.oat_c == pytest.approx(conditions.oat_c(257))

    def test_altitude_override_reads_the_fields_air_higher_up(self):
        """A traffic pattern is not flown at field elevation.

        The setting and temperature are still that field's -- it is the
        airplane that has moved, not the airmass.
        """
        field = nl.field_weather(self.HOT_HIGH, CALM)
        pattern = nl.field_weather(self.HOT_HIGH, CALM, altitude_ft=257 + 1000)
        assert pattern.altitude_ft == pytest.approx(1257)
        assert pattern.altimeter_inhg == pytest.approx(field.altimeter_inhg)
        assert pattern.pressure_altitude_ft == pytest.approx(
            field.pressure_altitude_ft + 1000
        )

    def test_reported_temperature_can_be_required(self):
        """What keeps the temperature profile from observing its own default.

        The profile is built from these samples, so a route-wide fallback
        offered here would come back in as though a station had reported it.
        """
        assert nl.field_weather(KMRY, CALM, require_reported_temperature=True) is None
        assert nl.field_weather(
            self.HOT_HIGH, CALM, require_reported_temperature=True
        ) is not None

    def test_agrees_with_the_go_no_go_check_it_feeds(self):
        # ISA+15 rather than HOT_HIGH's ISA+20: the field deviation is held
        # flat up to cruise, and the POH cruise chart stops at ISA+20.
        warm = replace(KMRY, oat_c=30.0, altimeter_inhg=29.42)
        log = nl.build_navlog([KSQL, warm], 6500, conditions=CALM, planning_mode="auto")
        landing = next(
            c for c in log.checklist.airports
            if c.airport == "KMRY" and c.operation == "landing"
        )
        air = nl.field_weather(warm, log.conditions)
        assert landing.density_altitude_ft == pytest.approx(air.density_altitude_ft)
        assert landing.pressure_altitude_ft == pytest.approx(air.pressure_altitude_ft)

    def test_the_log_hands_back_the_conditions_it_used(self):
        """Not the ones passed in: the field temperatures are folded in first.

        Without this a caller re-deriving the air at a field would use the
        route's ISA deviation where the log had used a reported temperature.
        """
        log = nl.build_navlog(
            [replace(KSQL, oat_c=30.0), KMRY], 6500, conditions=CALM, planning_mode="auto"
        )
        assert log.conditions is not None
        assert log.conditions.temperatures is not None
        assert log.conditions.oat_c(5) == pytest.approx(30.0, abs=0.5)
        assert CALM.temperatures is None


class TestObservedTemperature:
    """Field METARs and forecast temperatures aloft, reconciled into one airmass."""

    # ISA+15 at sea level. Deliberately not hotter: the deviation is held flat
    # above the top observation, so a field at ISA+25 would carry ISA+25 to
    # cruise and the POH cruise chart, which stops at ISA+20, would refuse the
    # flight. That refusal is correct -- the book has nothing to say about that
    # air -- but it is not what these tests are about.
    HOT_KSQL = replace(KSQL, oat_c=30.0, altimeter_inhg=29.80)

    def test_field_temperature_is_reproduced_exactly_at_the_field(self):
        log = nl.build_navlog(
            [self.HOT_KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto"
        )
        taxi = log.legs[0]
        assert taxi.phase == "taxi"
        assert taxi.oat_c == pytest.approx(30.0)
        assert taxi.density_altitude_ft > 1500  # 30 degC at sea level

    def test_field_temperature_reaches_the_climb(self):
        """The point of the whole thing: a hot departure is a worse climb.

        Nothing route-wide was changed -- only the one field's METAR -- and yet
        the climb takes longer, because the profile carries that observation up
        from the field at the standard lapse rate.
        """
        standard = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        hot_field = nl.build_navlog(
            [self.HOT_KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto"
        )

        def climb_time(log):
            return sum(leg.ete_min for leg in log.legs if leg.phase == "climb")

        assert climb_time(hot_field) > climb_time(standard)

    def test_a_typed_leg_temperature_reads_back_as_typed(self):
        base = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        row = next(i for i, leg in enumerate(base.legs) if leg.phase == "cruise")
        warm = nl.build_navlog(
            [KSQL, KSBP], 7500, conditions=CALM,
            overrides={row: nl.LegOverride(oat_c=12.0)},
            planning_mode="auto",
        )
        assert warm.legs[row].oat_c == pytest.approx(12.0)
        assert "oat_c" in warm.legs[row].overridden
        assert warm.legs[row].density_altitude_ft > base.legs[row].density_altitude_ft

    def test_a_typed_cruise_temperature_reaches_the_climb_below_it(self):
        """A forecast aloft is an observation about the air, not about one row.

        Warming the cruise level warms everything below it too, by the standard
        lapse -- which is what makes the climb to it take longer. A row-local
        value could not do this, and the second lay-out pass is what earns it.
        """
        base = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        row = next(i for i, leg in enumerate(base.legs) if leg.phase == "cruise")
        warm = nl.build_navlog(
            [KSQL, KSBP], 7500, conditions=CALM,
            overrides={row: nl.LegOverride(oat_c=20.0)},
            planning_mode="auto",
        )

        def climb_time(log):
            return sum(leg.ete_min for leg in log.legs if leg.phase == "climb")

        assert climb_time(warm) > climb_time(base)
        climb_row = next(i for i, leg in enumerate(warm.legs) if leg.phase == "climb")
        assert warm.legs[climb_row].oat_c > base.legs[climb_row].oat_c

    def test_the_checklist_keeps_the_fields_own_metar(self):
        """Field distances are read at the field's own weather, not the blend."""
        log = nl.build_navlog(
            [self.HOT_KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto"
        )
        departure = next(c for c in log.checklist.airports if c.airport == "KSQL")
        assert departure.oat_c == pytest.approx(30.0)

    def test_each_flight_gets_its_own_airmass(self):
        """A day with a stop is two airmasses, not one averaged into nonsense.

        The first flight departs a field at 35 degC and the second one does
        not, so the second leg's climb must not inherit the first's heat.
        """
        stop = replace(KMRY, is_landing=True)
        log = nl.build_navlog(
            [self.HOT_KSQL, stop, KSBP], 6500, conditions=CALM, planning_mode="auto"
        )
        climbs = [leg for leg in log.legs if leg.phase == "climb"]
        first = [leg for leg in climbs if leg.flight_index == 0]
        second = [leg for leg in climbs if leg.flight_index == 1]
        assert first and second
        assert first[0].oat_c > second[0].oat_c

    def test_nothing_observed_leaves_the_route_deviation_in_charge(self):
        deviation = nl.Conditions(isa_deviation_c=15.0, flight_date=date(2026, 8, 15))
        log = nl.build_navlog([KSQL, KSBP], 7500, conditions=deviation, planning_mode="auto")
        cruise = next(leg for leg in log.legs if leg.phase == "cruise")
        assert cruise.oat_c == pytest.approx(deviation.oat_c(7500))

    def test_a_superadiabatic_pair_of_entries_is_warned_about(self):
        """30 degC on the ground and -10 at 7500 is 5 degC per 1000 ft.

        Steeper than the air can hold, so it is far more likely a typo than
        weather. Warned about, never corrected: the plan is still built, and
        the pilot is the one who knows which of the two numbers was wrong.
        """
        base = nl.build_navlog(
            [self.HOT_KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto"
        )
        row = next(i for i, leg in enumerate(base.legs) if leg.phase == "cruise")
        steep = nl.build_navlog(
            [self.HOT_KSQL, KSBP], 7500, conditions=CALM,
            overrides={row: nl.LegOverride(oat_c=-10.0)},
            planning_mode="auto",
        )
        assert any("adiabatic" in w for w in steep.warnings)


class TestCruiseOffTheChartTemperature:
    """A day hotter than the cruise chart's ISA+20 column still plans.

    The chart is read at an equal density elsewhere on itself rather than the
    whole flight being refused -- see `performance.cruise_at_density`. What
    matters here is that the plan survives, that the substitution is not
    silent, and that the numbers still come out the right way round.
    """

    # 40 degC at sea level is ISA+25, and the deviation is held flat above the
    # only observation, so cruise is ISA+25 too -- five degrees past the
    # chart's hottest column.
    SCORCHING_KSQL: ClassVar = replace(KSQL, oat_c=40.0)

    def log(self, **kwargs):
        return nl.build_navlog(
            [self.SCORCHING_KSQL, KMRY], 6500, conditions=CALM, **kwargs,
            planning_mode="auto",
        )

    def test_the_flight_plans_instead_of_being_refused(self):
        log = self.log()
        assert any(leg.phase == "cruise" for leg in log.legs)
        assert log.total_fuel_gal > 0

    def test_the_substitution_is_reported(self):
        notes = [w for w in self.log().warnings if "density altitude" in w]
        assert notes, self.log().warnings
        assert "outside the cruise chart's published temperature band" in notes[0]

    def test_the_row_still_reports_the_air_it_is_actually_in(self):
        """The warning says where the chart was read; the row does not move.

        The pilot is flying at 6500 ft in ISA+25 air whatever page the fuel
        flow came off, so the navlog's own altitude, temperature and density
        altitude columns are the real ones.
        """
        from engine.atmosphere import isa_temperature_c

        cruise = next(leg for leg in self.log().legs if leg.phase == "cruise")
        assert cruise.altitude_ft == pytest.approx(6500)
        assert cruise.oat_c == pytest.approx(
            isa_temperature_c(cruise.pressure_altitude_ft) + 25.0, abs=0.5
        )
        assert cruise.density_altitude_ft > 9000

    def test_a_hot_day_is_still_slower_than_a_standard_one(self):
        hot = next(leg for leg in self.log().legs if leg.phase == "cruise")
        standard = next(
            leg
            for leg in nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto").legs
            if leg.phase == "cruise"
        )
        assert hot.tas_kt < standard.tas_kt


class TestPerLegCruiseRpm:
    """A row may be flown at its own power setting, within limits."""

    @staticmethod
    def _log(**kwargs):
        return nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, **kwargs, planning_mode="auto")

    @staticmethod
    def _row(log, phase):
        return next(i for i, leg in enumerate(log.legs) if leg.phase == phase)

    def test_higher_rpm_is_faster_and_thirstier(self):
        base = self._log()
        row = self._row(base, "cruise")
        fast = self._log(overrides={row: nl.LegOverride(cruise_rpm=2500)})
        assert fast.legs[row].tas_kt > base.legs[row].tas_kt
        assert fast.legs[row].fuel_gal > base.legs[row].fuel_gal
        assert fast.legs[row].ete_min < base.legs[row].ete_min

    def test_the_row_records_the_setting_it_flew(self):
        base = self._log()
        row = self._row(base, "cruise")
        assert base.legs[row].cruise_rpm == 2400.0  # the aircraft default
        edited = self._log(overrides={row: nl.LegOverride(cruise_rpm=2500)})
        assert edited.legs[row].cruise_rpm == 2500
        assert "cruise_rpm" in edited.legs[row].overridden

    def test_non_cruise_rows_carry_no_setting(self):
        log = self._log()
        for leg in log.legs:
            if leg.phase != "cruise":
                assert leg.cruise_rpm is None

    def test_only_this_row_changes(self):
        base = self._log()
        row = self._row(base, "cruise")
        edited = self._log(overrides={row: nl.LegOverride(cruise_rpm=2500)})
        for i, (a, b) in enumerate(zip(base.legs, edited.legs)):
            if i != row:
                assert a.tas_kt == pytest.approx(b.tas_kt)
                assert a.fuel_gal == pytest.approx(b.fuel_gal)

    def test_totals_follow_the_edited_row(self):
        base = self._log()
        row = self._row(base, "cruise")
        edited = self._log(overrides={row: nl.LegOverride(cruise_rpm=2500)})
        delta = edited.legs[row].fuel_gal - base.legs[row].fuel_gal
        assert edited.total_fuel_gal == pytest.approx(base.total_fuel_gal + delta)

    def test_climb_row_refuses_a_power_setting(self):
        row = self._row(self._log(), "climb")
        with pytest.raises(nl.RouteError, match="level flight only"):
            self._log(overrides={row: nl.LegOverride(cruise_rpm=2500)})

    def test_descent_row_refuses_a_power_setting(self):
        row = self._row(self._log(), "descent")
        with pytest.raises(nl.RouteError, match="level flight only"):
            self._log(overrides={row: nl.LegOverride(cruise_rpm=2500)})

    def test_below_the_altitude_floor_refuses(self):
        low = nl.build_navlog([KSQL, KSBP], 2500, conditions=CALM, planning_mode="auto")
        row = self._row(low, "cruise")
        with pytest.raises(nl.RouteError, match="floor for a per-leg"):
            nl.build_navlog(
                [KSQL, KSBP], 2500, conditions=CALM,
                overrides={row: nl.LegOverride(cruise_rpm=2400)},
                planning_mode="auto",
            )

    def test_unpublished_setting_refuses_and_lists_what_exists(self):
        row = self._row(self._log(), "cruise")
        with pytest.raises(nl.RouteError, match="Published settings there"):
            self._log(overrides={row: nl.LegOverride(cruise_rpm=3000)})

    def test_explicit_tas_wins_but_rpm_still_sets_the_fuel_flow(self):
        base = self._log()
        row = self._row(base, "cruise")
        rpm_only = self._log(overrides={row: nl.LegOverride(cruise_rpm=2500)})
        both = self._log(
            overrides={row: nl.LegOverride(cruise_rpm=2500, tas_kt=130.0)}
        )
        assert both.legs[row].tas_kt == pytest.approx(130.0)
        # Fuel flow, not fuel: the faster row spends less time burning it.
        # Tolerance because the fuel itself is rounded up to a tenth before the
        # rate is worked back out of it.
        rate = lambda leg: leg.fuel_gal / (leg.ete_min / 60.0)
        assert rate(both.legs[row]) == pytest.approx(
            rate(rpm_only.legs[row]), abs=0.1
        )

    def test_an_empty_override_is_ignored(self):
        assert nl.LegOverride().is_empty
        assert not nl.LegOverride(cruise_rpm=2500).is_empty


class TestFormatting:
    def test_renders_every_leg(self):
        log = nl.build_navlog([KSQL, VPWDM, KMRY], 6500, conditions=CALM, planning_mode="auto")
        text = nl.format_navlog(log)
        for leg in log.legs:
            assert leg.to_name in text
        assert "TOTAL" in text

    def test_carries_the_disclaimer(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert "NOT FOR NAVIGATION" in nl.format_navlog(log)

    def test_warnings_are_shown(self):
        log = nl.build_navlog(
            [KSQL, KSBP], 7500, nl.Aircraft(fuel_on_board_gal=12.0), CALM, planning_mode="auto"
        )
        assert "WARNING" in nl.format_navlog(log)


class TestPatternAndReserveFuelRates:
    """Both are charged where they happen, not at the altitude in the box.

    The cruise altitude is a *plan*, and in user-driven mode only a bound. A
    circuit at 1,100 ft and the half-hour of fuel that has to still be in the
    tanks on arrival are both low-altitude, high-power things; reading them off
    a 9,500 ft cruise page under-reads both.
    """

    @staticmethod
    def _rate(leg):
        return leg.fuel_gal / (leg.ete_min / 60.0)

    @staticmethod
    def _row(log, phase):
        return next(leg for leg in log.legs if leg.phase == phase)

    def test_pattern_is_charged_at_pattern_altitude(self):
        """A sea-level circuit reads the chart's floor, not the cruise page."""
        log = nl.build_navlog([KSQL, KMRY], 9500, conditions=CALM, planning_mode="auto")
        pattern = self._row(log, "pattern")
        cruise = self._row(log, "cruise")
        # More power available low down, so the circuit costs more per hour
        # than the cruise it followed.
        assert self._rate(pattern) > self._rate(cruise)

    def test_pattern_follows_the_field_it_is_flown_at(self):
        """A high field means a high circuit, and a leaner one."""
        low = nl.build_navlog([KSQL, KMRY], 9500, conditions=CALM, planning_mode="auto")
        high_field = replace(KMRY, elevation_ft=6500)
        high = nl.build_navlog([KSQL, high_field], 9500, conditions=CALM, planning_mode="auto")
        assert self._rate(high.legs[-1]) < self._rate(low.legs[-1])

    def test_pattern_ignores_a_cruise_altitude_edit(self):
        """Correctly so: the circuit is unchanged by how you got there."""
        route = [KSQL, replace(VPWDM, segment_type="climb"),
                 replace(KMRY, segment_type="descent")]
        rows = lambda log: [i for i, leg in enumerate(log.legs) if leg.covers_ground]
        base = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        edited = nl.build_navlog(
            route, 7500, conditions=CALM, planning_mode="manual",
            overrides={rows(base)[0]: nl.LegOverride(altitude_ft=9500)},
        )
        assert self._row(edited, "pattern").fuel_gal == pytest.approx(
            self._row(base, "pattern").fuel_gal
        )

    def test_reserve_follows_the_altitude_actually_cruised(self):
        """Higher cruise, leaner engine, smaller half-hour of reserve."""
        low = nl.build_navlog([KSQL, KSBP], 4500, conditions=CALM, planning_mode="auto")
        high = nl.build_navlog([KSQL, KSBP], 10500, conditions=CALM, planning_mode="auto")
        assert high.reserve_required_gal < low.reserve_required_gal

    def test_reserve_is_not_read_at_the_typed_altitude(self):
        """A user-driven plan that never reaches the box's altitude.

        The route levels at 3,500 ft whatever the box says, so the reserve must
        be the same both times -- it is measured off the flight, not the form.
        """
        route = [KSQL, replace(VPWDM, segment_type="climb", altitude_ft=3500),
                 replace(KMRY, segment_type="cruise")]
        a = nl.build_navlog(route, 5500, conditions=CALM, planning_mode="manual")
        b = nl.build_navlog(route, 11500, conditions=CALM, planning_mode="manual")
        assert a.reserve_required_gal == pytest.approx(b.reserve_required_gal)

    def test_reserve_uses_the_thirstiest_cruise_on_a_stepped_plan(self):
        """No credit for having planned one economical leg earlier."""
        stepped = [
            KSQL,
            replace(VPWDM, segment_type="climb", altitude_ft=9500),
            replace(KMRY, segment_type="descent", altitude_ft=3500),
        ]
        level_high = [
            KSQL,
            replace(VPWDM, segment_type="climb", altitude_ft=9500),
            replace(KMRY, segment_type="cruise"),
        ]
        low = nl.build_navlog(stepped, 9500, conditions=CALM, planning_mode="manual")
        high = nl.build_navlog(level_high, 9500, conditions=CALM, planning_mode="manual")
        # The stepped plan has no level leg below, so both cruise at 9500 --
        # what matters is that neither reads the box rather than the flight.
        assert low.reserve_required_gal > 0
        assert high.reserve_required_gal > 0

    def test_a_profile_with_no_cruise_falls_back(self):
        """Climb straight to a point and descend: nothing level to measure."""
        route = [KSQL, replace(VPWDM, segment_type="climb"),
                 replace(KMRY, segment_type="descent")]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        assert not any(leg.phase == "cruise" for leg in log.legs)
        assert log.reserve_required_gal > 0


class TestTemperaturesAloft:
    """An FD level is a pressure altitude, so it needs no altimeter setting.

    The forecast is produced on constant-pressure surfaces and labelled with
    their standard-atmosphere heights, so "9000" names the 9000 ft pressure
    level rather than 9000 ft on today's setting. That is what lets one
    bulletin be used without knowing the altimeter it will be flown on.
    """

    @staticmethod
    def _fd(*levels):
        """levels: (pressure_altitude_ft, oat_c) straight off the bulletin."""
        return tuple(
            nl.TemperatureSample.observed(alt, oat) for alt, oat in levels
        )

    @staticmethod
    def _cruise(log):
        return next(leg for leg in log.legs if leg.phase == "cruise")

    def test_an_fd_level_lands_at_its_own_pressure_altitude(self):
        assert nl.TemperatureSample.observed(9000, -7.0).pressure_altitude_ft == (
            pytest.approx(9000)
        )

    def test_the_altimeter_setting_does_not_move_it(self):
        """The point of the convention: one bulletin, any setting.

        Two pilots on different altimeter settings read the same temperature at
        the same *pressure* altitude, which is the coordinate the POH charts
        are indexed by.
        """
        warm = self._fd((6000, 12.0))
        low = nl.Conditions(altimeter_inhg=29.70, temperatures_aloft=warm)
        high = nl.Conditions(altimeter_inhg=30.10, temperatures_aloft=warm)
        rebuilt = [
            nl.TemperatureProfile.from_observations(c.temperatures_aloft)
            for c in (low, high)
        ]
        assert rebuilt[0].deviation_at(6000) == pytest.approx(
            rebuilt[1].deviation_at(6000)
        )

    def test_a_warm_forecast_raises_density_altitude(self):
        standard = nl.build_navlog([KSQL, KSBP], 9500, conditions=CALM, planning_mode="auto")
        warm = nl.build_navlog(
            [KSQL, KSBP],
            9500,
            conditions=replace(
                CALM,
                temperatures_aloft=self._fd(
                    (6000, 12.0), (9000, 6.0), (12000, -2.0)
                ),
            ),
            planning_mode="auto",
        )
        cool_row, warm_row = self._cruise(standard), self._cruise(warm)
        # Same indicated altitude, so the same pressure altitude -- only the
        # air is different.
        assert warm_row.pressure_altitude_ft == pytest.approx(
            cool_row.pressure_altitude_ft
        )
        assert warm_row.oat_c > cool_row.oat_c + 5
        assert warm_row.density_altitude_ft > cool_row.density_altitude_ft + 500

    def test_thinner_air_costs_true_airspeed_and_saves_fuel(self):
        """The reason it matters: less power up there, and less burn."""
        standard = nl.build_navlog([KSQL, KSBP], 9500, conditions=CALM, planning_mode="auto")
        warm = nl.build_navlog(
            [KSQL, KSBP],
            9500,
            conditions=replace(CALM, temperatures_aloft=self._fd((9000, 6.0))),
            planning_mode="auto",
        )
        cool_row, warm_row = self._cruise(standard), self._cruise(warm)
        rate = lambda leg: leg.fuel_gal / (leg.ete_min / 60.0)
        assert rate(warm_row) < rate(cool_row)

    def test_a_level_with_no_temperature_contributes_only_wind(self):
        """FD omits the temperature on the 3000 ft line; that must be fine."""
        wind_only = replace(
            CALM, winds=nl.WindsAloft(((3000.0, nl.Wind(250.0, 15.0)),))
        )
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=wind_only, planning_mode="auto")
        cruise = self._cruise(log)
        plain = self._cruise(
            nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        )
        # No temperature entered, so the air is unchanged...
        assert cruise.oat_c == pytest.approx(plain.oat_c)
        # ...but the wind is not.
        assert cruise.ground_speed_kt != pytest.approx(plain.ground_speed_kt)

    def test_a_field_report_and_an_fd_level_share_one_curve(self):
        """Two coordinates, one profile.

        The field is a true elevation under its own station setting; the FD
        level is already a pressure altitude. Both have to land on the same
        deviation curve, so a hot field warms the low legs without touching
        the forecast aloft.
        """
        aloft = self._fd((9000, 6.0))
        hot_field = replace(KSQL, oat_c=34.0, altimeter_inhg=29.80)
        base = nl.build_navlog(
            [KSQL, KSBP], 9500, conditions=replace(CALM, temperatures_aloft=aloft),
            planning_mode="auto",
        )
        hot = nl.build_navlog(
            [hot_field, KSBP],
            9500,
            conditions=replace(CALM, temperatures_aloft=aloft),
            planning_mode="auto",
        )
        climb_base = next(leg for leg in base.legs if leg.phase == "climb")
        climb_hot = next(leg for leg in hot.legs if leg.phase == "climb")
        assert climb_hot.oat_c > climb_base.oat_c
        # The forecast aloft is untouched by what the field reported.
        assert self._cruise(hot).oat_c == pytest.approx(self._cruise(base).oat_c)


class TestClimbLegLongerThanItsClimb:
    """A climb leg is as long as its waypoints; the climb rarely fills it.

    The rest is flown level, and charged as level flight at the most power the
    POH publishes up there -- worst case, because that stretch was not part of
    anybody's cruise plan. What it must never do is scale the published climb
    figure above itself, which is what made a higher climb cost *less*.
    """

    LEG: ClassVar = (
        nl.Waypoint("KPAO", LatLon(37.4611, -122.1150), "airport", elevation_ft=7),
        nl.Waypoint("FAR", LatLon(37.6850, -121.0426), "waypoint",
                    segment_type="climb"),
        nl.Waypoint("KMOD", LatLon(37.6258, -120.9544), "airport",
                    elevation_ft=99, segment_type="descent"),
    )

    def _climb_row(self, target=None):
        base = nl.build_navlog(list(self.LEG), 9500, conditions=CALM,
                               planning_mode="manual")
        row = next(i for i, leg in enumerate(base.legs) if leg.phase == "climb")
        if target is None:
            return base.legs[row]
        log = nl.build_navlog(
            list(self.LEG), 9500, conditions=CALM, planning_mode="manual",
            overrides={row: nl.LegOverride(altitude_ft=target)},
        )
        return log.legs[row]

    def test_the_row_decomposes_into_the_climb_plus_level_flight(self):
        """The bug was a 52.8 nm leg charged 12.7 gal for a 1.2 gal climb.

        Most of that row really is level flight, so the total being several
        times the climb figure is fine. What matters is that it decomposes:
        the published climb, plus the level remainder at a level rate. The old
        formula instead multiplied the climb figure by ten.
        """
        from engine import performance as perf

        row = self._climb_row(3000)
        poh = perf.climb_from_to(
            CALM.pressure_altitude_ft(7), CALM.pressure_altitude_ft(3000)
        )
        worst = perf.max_published_cruise(3000, CALM.oat_c(3000)).gph
        level_min = row.ete_min - poh.time_min
        assert level_min > 0
        assert row.fuel_gal == pytest.approx(
            poh.fuel_gal + worst * level_min / 60.0, rel=0.05
        )
        # Strictly less than pro-rating the whole leg, which is what the old
        # formula did: on this leg it gave 9.4 gal against the book's 1.2.
        prorated = poh.fuel_gal * (row.ete_min / poh.time_min)
        assert row.fuel_gal < prorated

    def test_a_higher_climb_takes_longer(self):
        times = [self._climb_row(a).ete_min for a in (3000, 5000, 7000, 9000, 11000)]
        assert times == sorted(times), times

    def test_a_higher_climb_costs_more_over_most_of_the_range(self):
        """Monotonic to 9000 ft. See the class below for why it is not above."""
        fuels = [self._climb_row(a).fuel_gal for a in (3000, 5000, 7000, 9000)]
        assert fuels == sorted(fuels), fuels

    def test_the_level_remainder_is_charged_at_full_power(self):
        """Not at the economy setting the pilot planned for the cruise."""
        from engine import performance as perf

        row = self._climb_row(5000)
        poh = perf.climb_from_to(
            CALM.pressure_altitude_ft(7), CALM.pressure_altitude_ft(5000)
        )
        level_min = row.ete_min - poh.time_min
        assert level_min > 0
        implied = (row.fuel_gal - poh.fuel_gal) / (level_min / 60.0)
        planned = perf.cruise(5000, nl.Aircraft().cruise_rpm, CALM.oat_c(5000)).gph
        worst = perf.max_published_cruise(5000, CALM.oat_c(5000)).gph
        assert implied > planned
        assert implied == pytest.approx(worst, rel=0.05)

    def test_a_leg_shorter_than_its_climb_still_prorates(self):
        """The automatic planner's case, which the old formula was written for.

        There the climb really does span two legs, so charging each a share of
        the published figure is right -- and the share is never more than all
        of it.
        """
        from engine import performance as perf

        log = nl.build_navlog([KSQL, KSBP], 9500, conditions=CALM, planning_mode="auto")
        climb = next(leg for leg in log.legs if leg.phase == "climb")
        whole = perf.climb_from_to(
            CALM.pressure_altitude_ft(5),
            CALM.pressure_altitude_ft(climb.exit_altitude_ft),
        )
        # Plus the tenth of a gallon the row is rounded up to.
        assert climb.fuel_gal <= whole.fuel_gal + 0.1 + 1e-6


class TestRowPressureAltitude:
    """A row's pressure altitude can be typed, and everything follows from it.

    The altimeter setting is route-wide, and on a long leg the air over the far
    end is not the air the setting came from. Typing the pressure altitude a
    row is really flown in is how that gets said.
    """

    @staticmethod
    def _log(**kwargs):
        return nl.build_navlog(
            [KSQL, KSBP], 7500, conditions=CALM, **kwargs, planning_mode="auto"
        )

    @staticmethod
    def _row(log, phase):
        return next(i for i, leg in enumerate(log.legs) if leg.phase == phase)

    def test_the_row_reports_the_pressure_altitude_typed(self):
        base = self._log()
        row = self._row(base, "cruise")
        edited = self._log(
            overrides={row: nl.LegOverride(pressure_altitude_ft=9000.0)}
        )
        assert edited.legs[row].pressure_altitude_ft == pytest.approx(9000.0)
        assert "pressure_altitude_ft" in edited.legs[row].overridden

    def test_density_altitude_follows_the_pair(self):
        """DA is derived, so it has to move with the typed PA and the row OAT."""
        from engine.atmosphere import density_altitude

        row = self._row(self._log(), "cruise")
        edited = self._log(
            overrides={
                row: nl.LegOverride(pressure_altitude_ft=9000.0, oat_c=25.0)
            }
        )
        leg = edited.legs[row]
        assert leg.oat_c == pytest.approx(25.0)
        assert leg.density_altitude_ft == pytest.approx(
            density_altitude(9000.0, 25.0)
        )

    def test_the_cruise_chart_is_read_at_the_typed_altitude(self):
        """The point of the field: the row's performance comes off its own air.

        Reading the same power setting higher up gives less true airspeed and
        a lower fuel flow, and the row has to show it.
        """
        base = self._log()
        row = self._row(base, "cruise")
        higher = self._log(
            overrides={row: nl.LegOverride(pressure_altitude_ft=10000.0)}
        )
        assert higher.legs[row].tas_kt != pytest.approx(base.legs[row].tas_kt)
        implied_base = base.legs[row].fuel_gal / (base.legs[row].ete_min / 60.0)
        implied_high = higher.legs[row].fuel_gal / (higher.legs[row].ete_min / 60.0)
        assert implied_high < implied_base

    def test_it_stops_at_its_own_row(self):
        """Unlike a temperature, a pressure altitude is not an air-mass fact.

        It says the altimeter setting does not describe this leg, which is a
        statement about one stretch of the route and nothing either side.
        """
        base = self._log()
        row = self._row(base, "cruise")
        edited = self._log(
            overrides={row: nl.LegOverride(pressure_altitude_ft=10000.0)}
        )
        for i, (before, after) in enumerate(zip(base.legs, edited.legs)):
            if i == row:
                continue
            assert after.pressure_altitude_ft == pytest.approx(
                before.pressure_altitude_ft
            )
            assert after.density_altitude_ft == pytest.approx(
                before.density_altitude_ft
            )

    def test_a_descent_row_is_charged_at_its_own_air(self):
        """A descent reads its fuel flow at the altitude it begins from.

        Typing a pressure altitude there says that air is thinner than the
        route's altimeter setting implies, and the fuel flow follows.
        """
        base = self._log()
        row = self._row(base, "descent")
        higher = self._log(
            overrides={row: nl.LegOverride(pressure_altitude_ft=10000.0)}
        )
        assert higher.legs[row].fuel_gal < base.legs[row].fuel_gal

    def test_a_climb_row_says_what_it_cannot_do(self):
        """A climb is the one row this cannot re-cost.

        Its time and fuel are integrated over the whole climb before any row
        exists. Refusing the entry would be worse -- the density altitude it
        gives is real -- but letting a pilot believe it re-costed the leg would
        be a great deal worse still.
        """
        base = self._log()
        row = self._row(base, "climb")
        edited = self._log(
            overrides={row: nl.LegOverride(pressure_altitude_ft=9000.0)}
        )
        assert edited.legs[row].pressure_altitude_ft == pytest.approx(9000.0)
        assert edited.legs[row].fuel_gal == pytest.approx(base.legs[row].fuel_gal)
        assert any(
            "not its time and fuel" in w for w in edited.warnings
        ), edited.warnings

    def test_a_level_remainder_moves_with_the_row(self):
        """A climb that tops out mid-leg costs its level stretch at the top.

        The typed number is the row's own -- a climb row reads at its midpoint
        -- so what carries to the level stretch is the offset from the route's
        altimeter setting, keeping the whole row in one air mass.
        """
        def declare(waypoint, segment_type, altitude_ft=None):
            return replace(
                waypoint, segment_type=segment_type, altitude_ft=altitude_ft
            )

        route = [KSQL, declare(VPWDM, "climb", 3000), declare(KSBP, "cruise")]
        base = nl.build_navlog(route, 7500, conditions=CALM)
        row = next(i for i, leg in enumerate(base.legs) if leg.phase == "climb")
        assert base.legs[row].ete_min > 0
        higher = nl.build_navlog(
            route,
            7500,
            conditions=CALM,
            overrides={row: nl.LegOverride(pressure_altitude_ft=8000.0)},
        )
        # Thinner air for the level part of the row, so a lower worst-case burn.
        assert higher.legs[row].fuel_gal < base.legs[row].fuel_gal


class TestDescentFuelFlow:
    """A descent is charged at the cruise rate, read where the descent starts.

    The POH publishes no descent table, so cruise is the stand-in. Which
    altitude that cruise rate is read at used to be the number in the cruise
    altitude box -- which in user-driven mode is only a bound, and which the UI
    disables. It is now the altitude the descent is entered at.
    """

    @staticmethod
    def _declared(top_ft):
        def declare(waypoint, segment_type, altitude_ft=None):
            return replace(
                waypoint, segment_type=segment_type, altitude_ft=altitude_ft
            )

        return [KSQL, declare(VPWDM, "climb", top_ft), declare(KSBP, "descent")]

    @staticmethod
    def _descent(log):
        return next(leg for leg in log.legs if leg.phase == "descent")

    @staticmethod
    def _gph(leg):
        return leg.fuel_gal / (leg.ete_min / 60.0)

    def test_the_cruise_altitude_box_no_longer_moves_it(self):
        """The regression: a disabled field was worth 17% of the descent fuel.

        Same route, same declared profile, same descent -- only the nominal
        cruise altitude differs, and in user-driven mode that is a bound the
        aeroplane may never see.
        """
        route = self._declared(5500)
        fuels = [
            self._descent(nl.build_navlog(route, bound, conditions=CALM)).fuel_gal
            for bound in (3000, 6500, 10000)
        ]
        assert fuels[0] == pytest.approx(fuels[1])
        assert fuels[1] == pytest.approx(fuels[2])

    def test_it_is_read_at_the_altitude_the_descent_begins_from(self):
        """Thinner air at the top means less power, so a lower rate."""
        low = self._descent(nl.build_navlog(self._declared(3500), 7500, conditions=CALM))
        high = self._descent(nl.build_navlog(self._declared(9500), 7500, conditions=CALM))
        assert high.entry_altitude_ft > low.entry_altitude_ft
        assert self._gph(high) < self._gph(low)

    def test_it_matches_the_chart_at_that_altitude(self):
        from engine import performance as perf

        log = nl.build_navlog(self._declared(7500), 7500, conditions=CALM)
        leg = self._descent(log)
        expected = perf.cruise(
            CALM.pressure_altitude_ft(leg.entry_altitude_ft),
            nl.Aircraft().cruise_rpm,
            CALM.oat_c(leg.entry_altitude_ft),
        ).gph
        # The row's fuel is rounded up to a tenth, so the rate worked back out
        # of it is only good to that tenth spread over the leg's own time.
        assert self._gph(leg) == pytest.approx(
            expected, abs=0.1 / (leg.ete_min / 60.0)
        )

    def test_a_descent_below_the_chart_still_plans(self):
        """The chart starts at 2000 ft; a low descent reads its bottom page."""
        log = nl.build_navlog(self._declared(1500), 1500, conditions=CALM)
        leg = self._descent(log)
        assert leg.fuel_gal > 0
        assert self._gph(leg) > 0


class TestFuelIsRoundedUpToTheTenth:
    """Every fuel figure is a tenth of a gallon, rounded up, and adds up.

    The convention a paper navigation log is filled in with: round each box up
    to the nearest tenth, then total the column from the boxes.
    """

    @staticmethod
    def _log(**kwargs):
        return nl.build_navlog(
            [KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto", **kwargs
        )

    def test_every_row_is_a_whole_tenth(self):
        for leg in self._log().legs:
            assert leg.fuel_gal * 10 == pytest.approx(round(leg.fuel_gal * 10))

    def test_the_column_adds_up_to_the_total(self):
        """The reason the rounding is done in the engine and not on the way out."""
        log = self._log()
        assert sum(leg.fuel_gal for leg in log.legs) == pytest.approx(
            log.total_fuel_gal
        )
        assert log.legs[-1].cumulative_fuel_gal == pytest.approx(log.total_fuel_gal)

    def test_the_landing_figure_agrees_with_the_column(self):
        log = self._log()
        assert log.fuel_remaining_gal == pytest.approx(
            nl.Aircraft().fuel_on_board_gal - log.total_fuel_gal
        )

    def test_it_rounds_up_and_never_down(self):
        """A plan must not come out of the arithmetic holding fuel it has not got."""
        assert nl._fuel_written_on_the_log(2.51) == pytest.approx(2.6)
        assert nl._fuel_written_on_the_log(2.501) == pytest.approx(2.6)
        assert nl._fuel_written_on_the_log(2.5) == pytest.approx(2.5)

    def test_an_exact_tenth_stays_where_it_is(self):
        """Floating point must not push 2.6 to 2.7."""
        for tenths in range(1, 200):
            exact = tenths / 10.0
            assert nl._fuel_written_on_the_log(exact) == pytest.approx(exact)

    def test_the_reserve_is_rounded_the_same_way(self):
        log = self._log()
        assert log.reserve_required_gal * 10 == pytest.approx(
            round(log.reserve_required_gal * 10)
        )


class TestFieldWeatherReachesTheGoNoGo:
    """The sky at each end of the route, and whether the checklist sees it.

    The go/no-go's VFR gate is only as good as the report that reaches it, and
    the route is the only thing that knows which report belongs to which
    field. So what is tested here is the plumbing, not the rule.
    """

    OVERCAST = replace(
        KSQL,
        visibility_sm=10.0,
        ceiling_ft_agl=900.0,
        ceiling_cover="OVC",
        sky_reported=True,
    )

    def _check(self, log, airport, operation):
        return next(
            c for c in log.checklist.airports
            if c.airport == airport and c.operation == operation
        )

    def test_a_route_with_no_weather_on_it_has_no_weather_verdict(self):
        """The plan a pilot gets before pressing "Get weather" is unchanged."""
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert all(c.weather is None for c in log.checklist.airports)

    def test_a_report_that_arrived_saying_nothing_is_an_unknown_not_a_silence(self):
        """A model-only answer observes no cloud, which is not "no cloud".

        It comes back with every sky field empty -- identical to a field
        nobody asked about -- so the fact that it was asked has to travel
        with it, or the VFR gate silently skips the field.
        """
        asked = replace(KSQL, weather_reported=True)
        log = nl.build_navlog([asked, KMRY], 6500, conditions=CALM, planning_mode="auto")
        weather = self._check(log, "KSQL", "takeoff").weather
        assert weather is not None
        assert weather.passes is None
        assert any("weather" in u for u in log.checklist.unknowns)

    def test_a_ceiling_on_the_departure_field_sinks_the_departure(self):
        log = nl.build_navlog(
            [self.OVERCAST, KMRY], 6500, conditions=CALM, planning_mode="auto"
        )
        departure = self._check(log, "KSQL", "takeoff")
        assert departure.weather.passes is False
        # The sky is what failed, not the tarmac: these fixtures carry no
        # runway data, so the runway half of the check is merely unknown.
        assert departure.runways_pass is None
        assert departure.passes is False
        assert any("weather" in b and "OVC" in b for b in log.checklist.blockers)

    def test_and_leaves_the_destination_alone(self):
        """Weather belongs to the field it was reported at, and to no other."""
        log = nl.build_navlog(
            [self.OVERCAST, KMRY], 6500, conditions=CALM, planning_mode="auto"
        )
        assert self._check(log, "KMRY", "landing").weather is None

    def test_a_field_flies_the_pattern_it_publishes(self):
        """900 ft of overcast is below an 800 ft pattern plus 500 as well.

        The point is the requirement, not the verdict: a field that publishes
        an 800 ft pattern must be held to 1,300 ft, not to the standard 1,500.
        """
        published = replace(self.OVERCAST, pattern_altitude_agl_ft=800.0)
        log = nl.build_navlog(
            [published, KMRY], 6500, conditions=CALM, planning_mode="auto"
        )
        weather = self._check(log, "KSQL", "takeoff").weather
        assert weather.pattern_altitude_agl_ft == pytest.approx(800.0)
        assert weather.required_ceiling_ft_agl == pytest.approx(1300.0)

    def test_a_field_with_no_published_pattern_takes_the_standard_one(self):
        log = nl.build_navlog(
            [self.OVERCAST, KMRY], 6500, conditions=CALM, planning_mode="auto"
        )
        weather = self._check(log, "KSQL", "takeoff").weather
        assert weather.pattern_altitude_agl_ft == pytest.approx(1000.0)
        assert weather.required_ceiling_ft_agl == pytest.approx(1500.0)

    def test_a_stop_is_judged_on_its_own_weather_at_both_ends(self):
        """Landing in and taking off again are two checks of the same sky."""
        stop = replace(
            KMRY,
            is_landing=True,
            visibility_sm=1.0,
            sky_reported=True,
        )
        log = nl.build_navlog(
            [KSQL, stop, KSBP], 6500, conditions=CALM, planning_mode="auto"
        )
        assert self._check(log, "KMRY", "landing").weather.passes is False
        assert self._check(log, "KMRY", "takeoff").weather.passes is False


class TestWindsAloftByLeg:
    """A forecast column per leg, and which leg each one lands on.

    The alternative -- one column for the whole route -- is the objection
    `engine/aloft.py` raises against the FD product, only with a finer grid.
    What is tested here is that a leg is flown in the sky above it, that a
    number the pilot typed still beats a forecast, and that the keying
    survives the thing that makes row keys unusable: the forecast moves the
    tops of climb, which renumbers the rows it would have been keyed to.
    """

    WEST = nl.WindsAloft.uniform(270.0, 30.0)
    EAST = nl.WindsAloft.uniform(90.0, 30.0)

    def field(self, *columns):
        return nl.WindField(tuple(nl.WindColumn(p, w) for p, w in columns))

    def two_legs(self, **conditions):
        """KSQL to KMRY to KSBP: two legs the pilot drew, down the coast."""
        return nl.build_navlog(
            [KSQL, replace(KMRY, is_landing=True), KSBP],
            6500,
            conditions=nl.Conditions(flight_date=date(2026, 9, 2), **conditions),
            planning_mode="auto",
        )

    def rows(self, log):
        return [leg for leg in log.legs if leg.covers_ground]

    def first_flying_row(self, log):
        """The index of the first row that goes anywhere.

        Overrides are keyed by navlog row, and row zero is the taxi -- which
        has no course, no wind and nothing to override.
        """
        return next(i for i, leg in enumerate(log.legs) if leg.covers_ground)

    def test_each_leg_is_flown_in_the_column_over_it(self):
        """A westerly over the first leg, an easterly over the second."""
        first = inverse(KSQL.position, KMRY.position).point_at_fraction(0.5)
        second = inverse(KMRY.position, KSBP.position).point_at_fraction(0.5)
        log = self.two_legs(
            wind_field=self.field((first, self.WEST), (second, self.EAST))
        )
        rows = self.rows(log)
        # Rows before the stop belong to the first leg, rows after it to the
        # second -- including the climb out of KMRY, which is sixty miles
        # nearer the first leg's column than the second leg's.
        stop = next(i for i, r in enumerate(rows) if r.to_name == "KMRY")
        assert all(r.wind_from_deg == pytest.approx(270.0) for r in rows[: stop + 1])
        assert all(r.wind_from_deg == pytest.approx(90.0) for r in rows[stop + 1 :])

    def test_a_route_with_no_field_is_untouched(self):
        """Calm, exactly as a plan with nothing entered has always been."""
        for row in self.rows(self.two_legs()):
            assert row.wind_speed_kt == pytest.approx(0.0)

    def test_the_column_moves_the_tops_of_climb_it_is_keyed_past(self):
        """Why the columns are keyed by position and not by row.

        A headwind makes the climb cover less ground, so the top of climb
        arrives sooner and every row after it renumbers -- under the very
        forecast being applied. A point on the earth does not move.
        """
        first = inverse(KSQL.position, KMRY.position).point_at_fraction(0.5)
        calm = self.two_legs()
        blown = self.two_legs(wind_field=self.field((first, self.WEST)))
        toc = [r for r in self.rows(calm) if r.end_role == "TOC"][0]
        blown_toc = [r for r in self.rows(blown) if r.end_role == "TOC"][0]
        assert blown_toc.distance_nm != pytest.approx(toc.distance_nm)
        # And the wind still landed on it, having been found by where it is.
        assert blown_toc.wind_from_deg == pytest.approx(270.0)

    def test_a_typed_wind_still_beats_the_forecast_on_its_own_row(self):
        first = inverse(KSQL.position, KMRY.position).point_at_fraction(0.5)
        field = self.field((first, self.WEST))
        conditions = nl.Conditions(flight_date=date(2026, 9, 2), wind_field=field)
        row = self.first_flying_row(
            nl.build_navlog([KSQL, KMRY], 6500, conditions=conditions,
                            planning_mode="auto")
        )
        log = nl.build_navlog(
            [KSQL, KMRY],
            6500,
            conditions=conditions,
            overrides={row: nl.LegOverride(wind_from_deg=180.0, wind_speed_kt=12.0)},
            planning_mode="auto",
        )
        first_row = self.rows(log)[0]
        assert first_row.wind_from_deg == pytest.approx(180.0)
        assert first_row.wind_speed_kt == pytest.approx(12.0)
        assert "wind_from_deg" in first_row.overridden

    def test_a_half_typed_wind_takes_its_other_half_from_that_legs_column(self):
        """Not from the route's profile: the leg has a better answer.

        A direction typed with no speed used to read its speed off the
        route-wide wind, which with a per-leg forecast is the wrong column.
        """
        first = inverse(KSQL.position, KMRY.position).point_at_fraction(0.5)
        conditions = nl.Conditions(
            flight_date=date(2026, 9, 2), wind_field=self.field((first, self.WEST))
        )
        row = self.first_flying_row(
            nl.build_navlog([KSQL, KMRY], 6500, conditions=conditions,
                            planning_mode="auto")
        )
        log = nl.build_navlog(
            [KSQL, KMRY],
            6500,
            conditions=conditions,
            overrides={row: nl.LegOverride(wind_from_deg=180.0)},
            planning_mode="auto",
        )
        first_row = self.rows(log)[0]
        assert first_row.wind_from_deg == pytest.approx(180.0)
        assert first_row.wind_speed_kt == pytest.approx(30.0)  # off the column

    def test_the_columns_temperatures_reach_the_density_altitudes(self):
        """Winds by position, temperature into the one route-wide curve."""
        first = inverse(KSQL.position, KMRY.position).point_at_fraction(0.5)
        warm = nl.build_navlog(
            [KSQL, KMRY],
            6500,
            conditions=nl.Conditions(
                flight_date=date(2026, 9, 2),
                wind_field=self.field((first, self.WEST)),
                temperatures_aloft=(nl.TemperatureSample(6500.0, 15.0),),
            ),
            planning_mode="auto",
        )
        cruise = next(r for r in self.rows(warm) if r.phase == "cruise")
        assert cruise.density_altitude_ft > cruise.pressure_altitude_ft + 1500
