"""Vertical profile tests.

The invariant everything else rests on: after `resolve_route`, every leg has a
concrete segment type and `build_segments` emits **exactly one segment per
leg**. That is what makes navlog rows correspond 1:1 to legs, and therefore
what makes a row-indexed override land where the pilot aimed it.
"""

from dataclasses import replace
from datetime import date
from itertools import pairwise

import pytest

from engine import navlog as nl
from engine import profile as pr
from engine.geo import LatLon

KSQL = nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5)
KMRY = nl.Waypoint("KMRY", LatLon(36.5870, -121.8429), "airport", elevation_ft=257)
KSBP = nl.Waypoint("KSBP", LatLon(35.2368, -120.6424), "airport", elevation_ft=212)
VPWDM = nl.Waypoint("VPWDM", LatLon(37.2000, -122.0500), "vfr_waypoint")
# Deliberately close to KMRY -- about 7 nm -- for the cases where a leg is too
# short to fly the altitude change asked of it.
NEAR = nl.Waypoint("NEAR", LatLon(36.7000, -121.9000), "vfr_waypoint")

CALM = nl.Conditions(flight_date=date(2026, 8, 15))


def declare(waypoint, segment_type, altitude_ft=None):
    return replace(waypoint, segment_type=segment_type, altitude_ft=altitude_ft)


def resolve(route, mode="auto", altitude=7500, conditions=CALM, aircraft=None):
    return pr.resolve_route(
        route,
        mode=mode,
        cruise_altitude_ft=altitude,
        departure_elevation=route[0].elevation_ft or 0.0,
        destination_elevation=route[-1].elevation_ft or 0.0,
        aircraft=aircraft or nl.Aircraft(),
        conditions=conditions,
        names=pr.PhaseNamer(),
        warnings=[],
    )


def segments(route, conditions=CALM, aircraft=None):
    aircraft = aircraft or nl.Aircraft()
    point = nl.perf.cruise(
        conditions.pressure_altitude_ft(7500),
        aircraft.cruise_rpm,
        conditions.oat_c(7500),
    )
    return pr.build_segments(
        route,
        departure_elevation=route[0].elevation_ft or 0.0,
        aircraft=aircraft,
        conditions=conditions,
        cruise_point=point,
    )


class TestResolution:
    def test_automatic_expansion_inserts_toc_and_tod(self):
        resolved = resolve([KSQL, KMRY])
        generated = [w for w in resolved if w.generated]
        assert [w.name for w in generated] == ["TOC", "TOD"]
        assert all(w.kind == "phase" for w in generated)

    def test_every_leg_is_concrete_after_resolving(self):
        for route in ([KSQL, KMRY], [KSQL, VPWDM, KMRY], [KSQL, KSBP]):
            resolved = resolve(route)
            assert all(
                w.segment_type in pr.CONCRETE_SEGMENT_TYPES for w in resolved[1:]
            ), [w.segment_type for w in resolved]

    def test_resolution_is_idempotent(self):
        """Re-planning must not stack a second TOC beside the first."""
        once = resolve([KSQL, KMRY])
        twice = resolve(once)
        assert [w.name for w in once] == [w.name for w in twice]
        assert [w.segment_type for w in once] == [w.segment_type for w in twice]
        assert len([w for w in twice if w.generated]) == 2

    def test_generated_points_are_stripped_before_re_expanding(self):
        resolved = resolve([KSQL, KMRY])
        assert [w.name for w in pr.strip_generated(resolved)] == ["KSQL", "KMRY"]

    def test_expansion_does_not_write_altitudes_onto_the_pilots_waypoints(self):
        """Writing one back would read as a crossing restriction next time,
        moving the TOD and letting the route drift on every re-plan."""
        resolved = resolve([KSQL, KMRY])
        for waypoint in resolved:
            if not waypoint.generated:
                assert waypoint.altitude_ft is None

    def test_expansion_does_write_the_type_back(self):
        """That is the hand-off: an automatic pass leaves a route to edit."""
        resolved = resolve([KSQL, KMRY])
        assert resolved[-1].segment_type == "descent"

    def test_an_edited_generated_point_survives(self):
        """Clearing `generated` makes it the pilot's, so the strip keeps it."""
        resolved = resolve([KSQL, KMRY])
        adopted = [
            replace(w, generated=False, kind="waypoint") if w.generated else w
            for w in resolved
        ]
        assert len(pr.strip_generated(adopted)) == len(adopted)

    def test_manual_mode_refuses_an_undeclared_leg(self):
        with pytest.raises(nl.RouteError, match="still automatic"):
            resolve([KSQL, KMRY], mode="manual")

    def test_manual_mode_names_the_waypoint_that_needs_a_type(self):
        route = [KSQL, declare(VPWDM, "climb"), KMRY]
        with pytest.raises(nl.RouteError, match="KMRY"):
            resolve(route, mode="manual")

    def test_manual_mode_inserts_nothing(self):
        route = [KSQL, declare(VPWDM, "climb"), declare(KMRY, "descent")]
        assert resolve(route, mode="manual") == route

    def test_unknown_mode_refuses(self):
        with pytest.raises(nl.RouteError, match="planning mode"):
            resolve([KSQL, KMRY], mode="sideways")


class TestOneSegmentPerLeg:
    def test_manual(self):
        route = [KSQL, declare(VPWDM, "climb"), declare(KMRY, "descent")]
        assert len(segments(route)) == len(route) - 1

    def test_automatic(self):
        resolved = resolve([KSQL, KMRY])
        assert len(segments(resolved)) == len(resolved) - 1

    def test_the_profile_is_continuous(self):
        """Each leg starts where the one before it ended, by construction."""
        resolved = resolve([KSQL, VPWDM, KMRY])
        built = segments(resolved)
        for before, after in pairwise(built):
            assert after.entry_altitude_ft == pytest.approx(before.exit_altitude_ft)


class TestComputedAltitudes:
    def test_a_climb_with_no_altitude_climbs_as_high_as_it_gets(self):
        route = [KSQL, declare(KMRY, "climb")]
        built = segments(route)
        assert built[0].phase == "climb"
        assert built[0].exit_altitude_ft > built[0].entry_altitude_ft + 1000

    def test_a_long_climb_stops_at_the_top_of_the_chart(self):
        """"As high as it gets" means as high as the POH will answer for."""
        route = [KSQL, declare(KMRY, "climb")]
        reached = segments(route)[0].exit_altitude_ft
        assert reached == pytest.approx(nl.perf.climb_table_ceiling_ft(), abs=200)

    def test_wind_moves_the_altitude_a_climb_reaches(self):
        """A headwind puts the aeroplane *higher* over the waypoint.

        Counterintuitive but right: the climb is an air-mass manoeuvre, so a
        headwind buys more minutes of climbing before the same point on the
        ground goes past, and a tailwind spends fewer. The leg has to be short
        enough that the climb does not simply top out -- over the 59 nm to KMRY
        a 172 reaches the chart ceiling into any wind worth flying in.
        """
        route = [KSQL, declare(VPWDM, "climb")]  # 21 nm, about 153 true
        calm = segments(route)[0].exit_altitude_ft
        assert calm < nl.perf.climb_table_ceiling_ft() - 500  # room to move

        def with_wind(from_deg):
            conditions = nl.Conditions(
                flight_date=date(2026, 8, 15),
                winds=nl.WindsAloft(((0.0, nl.Wind(from_deg, 30.0)),)),
            )
            return segments(route, conditions=conditions)[0].exit_altitude_ft

        assert with_wind(153.0) > calm  # on the nose
        assert with_wind(333.0) < calm  # up the tail

    def test_a_cruise_leg_holds_its_altitude(self):
        route = [KSQL, declare(VPWDM, "climb", 5500), declare(KMRY, "cruise")]
        built = segments(route)
        assert built[1].entry_altitude_ft == pytest.approx(5500)
        assert built[1].exit_altitude_ft == pytest.approx(5500)

    def test_a_descent_with_no_altitude_uses_the_configured_rate(self):
        route = [KSQL, declare(VPWDM, "climb", 6500), declare(KMRY, "descent")]
        built = segments(route)
        descent = built[-1]
        lost = descent.entry_altitude_ft - descent.exit_altitude_ft
        minutes = 60.0 * nl.inverse(VPWDM.position, KMRY.position).distance_nm / 100.0
        # 500 fpm is the default; allow a wide band because the ground speed
        # depends on the altitude band, which is what the solver is finding.
        assert lost == pytest.approx(500.0 * minutes, rel=0.5)

    def test_a_descent_too_long_for_its_leg_gets_as_low_as_it_can(self):
        """The bisection branch: the short-circuit does not cover this.

        A high, short final leg cannot reach the field at 500 fpm. The answer
        must be the deepest altitude that *does* fit, not "no descent at all"
        and not a descent that needs more room than the leg has.
        """
        route = [KSQL, declare(NEAR, "climb", 11500), declare(KMRY, "descent")]
        built = segments(route)
        descent = built[-1]
        assert descent.entry_altitude_ft == pytest.approx(11500)
        # It descends, but does not make the field.
        assert descent.exit_altitude_ft < 11500
        assert descent.exit_altitude_ft > 257

    def test_a_descent_stops_at_field_elevation(self):
        """A long descent levels off at the field rather than digging in."""
        route = [KSQL, declare(VPWDM, "climb", 3000), declare(KMRY, "descent")]
        assert segments(route)[-1].exit_altitude_ft == pytest.approx(257.0)

    def test_an_explicit_altitude_wins(self):
        route = [KSQL, declare(KMRY, "climb", 4500)]
        assert segments(route)[0].exit_altitude_ft == pytest.approx(4500)

    def test_a_contradiction_is_emitted_not_corrected(self):
        """A "climb" to a lower altitude is built as declared, for the checker."""
        route = [KSQL, declare(VPWDM, "climb", 6500), declare(KMRY, "climb", 3000)]
        built = segments(route)
        assert built[-1].phase == "climb"
        assert built[-1].exit_altitude_ft < built[-1].entry_altitude_ft


class TestBoundaryRoles:
    def test_a_generated_point_is_named_toc(self):
        built = segments(resolve([KSQL, KMRY]))
        climb = next(s for s in built if s.phase == "climb")
        assert climb.end.name == "TOC"
        assert climb.end_role == "TOC"

    def test_a_charted_point_keeps_its_name_and_gains_a_role(self):
        route = [KSQL, declare(VPWDM, "climb", 6500), declare(KMRY, "descent")]
        built = segments(route)
        assert built[0].end.name == "VPWDM"  # not renamed
        assert built[0].end_role == "TOC"
        assert nl.leg_label("VPWDM", "TOC") == "VPWDM (TOC)"

    def test_a_descent_marks_its_start_as_the_top_of_descent(self):
        route = [KSQL, declare(VPWDM, "climb", 6500), declare(KMRY, "descent")]
        built = segments(route)
        assert built[-1].start_role == "TOD"

    def test_consecutive_climbs_have_no_top_between_them(self):
        """A climb that carries on is not a top of climb."""
        route = [
            KSQL,
            declare(VPWDM, "climb", 3000),
            declare(KMRY, "climb", 6500),
        ]
        built = segments(route)
        assert built[0].end_role is None
        assert built[1].end_role == "TOC"

    def test_a_plain_name_is_not_doubled_up(self):
        assert nl.leg_label("TOC", "TOC") == "TOC"


class TestAutomaticModeRegression:
    """The automatic planner moved modules; it must still plan the same way."""

    def test_the_ordinary_route_is_climb_cruise_descent(self):
        log = nl.build_navlog([KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto")
        flown = [leg.phase for leg in log.legs if leg.covers_ground]
        assert flown == ["climb", "cruise", "descent"]

    def test_a_crossing_altitude_is_honoured(self):
        """Previously untested: a waypoint altitude caps the leg arriving at it."""
        low = replace(VPWDM, altitude_ft=3500)
        log = nl.build_navlog([KSQL, low, KMRY], 7500, conditions=CALM, planning_mode="auto")
        crossing = [
            leg for leg in log.legs if leg.covers_ground and leg.to_name == "VPWDM"
        ]
        assert crossing
        assert crossing[-1].exit_altitude_ft == pytest.approx(3500, abs=1.0)

    def test_a_crossing_altitude_persists_down_route(self):
        """It applies to the leg arriving, and to every leg after it."""
        low = replace(VPWDM, altitude_ft=3500)
        log = nl.build_navlog([KSQL, low, KMRY], 7500, conditions=CALM, planning_mode="auto")
        after = [
            leg
            for leg in log.legs
            if leg.covers_ground and leg.phase == "cruise" and leg.from_name == "VPWDM"
        ]
        for leg in after:
            assert leg.altitude_ft == pytest.approx(3500, abs=1.0)

    def test_total_distance_is_unchanged_by_inserting_points(self):
        from engine.geo import inverse

        direct = (
            inverse(KSQL.position, VPWDM.position).distance_nm
            + inverse(VPWDM.position, KMRY.position).distance_nm
        )
        log = nl.build_navlog([KSQL, VPWDM, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert log.total_distance_nm == pytest.approx(direct, rel=1e-6)

    def test_the_resolved_route_is_handed_back(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert [w.name for w in log.resolved_waypoints] == [
            "KSQL",
            "TOC",
            "TOD",
            "KMRY",
        ]

    def test_an_automatic_pass_hands_off_to_user_driven(self):
        """The point of writing TOC/TOD into the route: you can take it over.

        Adopting them -- clearing `generated` -- is what makes them survive the
        strip at the top of resolution. Without that step the switch to
        user-driven mode discards them and leaves nothing between departure and
        destination, which is the opposite of a hand-off.
        """
        auto = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        adopted = [
            replace(w, generated=False, kind="waypoint") if w.generated else w
            for w in auto.resolved_waypoints
        ]
        manual = nl.build_navlog(
            adopted, 6500, conditions=CALM, planning_mode="manual"
        )
        assert [leg.phase for leg in manual.legs if leg.covers_ground] == [
            "climb",
            "cruise",
            "descent",
        ]
        assert manual.total_fuel_gal == pytest.approx(auto.total_fuel_gal, rel=1e-3)

    def test_not_adopting_them_loses_them(self):
        """Documents the trap the UI has to avoid on the mode switch."""
        auto = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        manual = nl.build_navlog(
            list(auto.resolved_waypoints),
            6500,
            conditions=CALM,
            planning_mode="manual",
        )
        # Stripped back to the pilot's two airports: one leg, not three.
        assert len([leg for leg in manual.legs if leg.covers_ground]) == 1

    def test_replanning_the_resolved_route_is_stable(self):
        once = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        twice = nl.build_navlog(
            list(once.resolved_waypoints), 6500, conditions=CALM,
            planning_mode="auto",
        )
        assert [w.name for w in twice.resolved_waypoints] == [
            w.name for w in once.resolved_waypoints
        ]
        assert twice.total_fuel_gal == pytest.approx(once.total_fuel_gal)


class TestManualModeThroughBuildNavlog:
    def test_user_driven_is_the_default_mode(self):
        """No `planning_mode` means the pilot's declarations, not the planner's.

        Planning is a thing a pilot does leg by leg, so the profile they get
        without asking is the one they declared. Automatic is the shortcut,
        and it has to be asked for.
        """
        route = [KSQL, declare(VPWDM, "climb"), declare(KMRY, "descent")]
        log = nl.build_navlog(route, 7500, conditions=CALM)
        assert log.planning_mode == "manual"
        assert [leg.phase for leg in log.legs if leg.covers_ground] == [
            "climb",
            "descent",
        ]

    def test_the_default_refuses_an_undeclared_route(self):
        """The other half of the default: it does not quietly plan for you."""
        with pytest.raises(nl.RouteError, match="user-driven planning needs"):
            nl.build_navlog([KSQL, VPWDM, KMRY], 7500, conditions=CALM)

    def test_a_declared_route_builds(self):
        route = [KSQL, declare(VPWDM, "climb"), declare(KMRY, "descent")]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        flown = [leg.phase for leg in log.legs if leg.covers_ground]
        assert flown == ["climb", "descent"]

    def test_one_row_per_leg(self):
        route = [KSQL, declare(VPWDM, "climb"), declare(KMRY, "descent")]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        assert len([leg for leg in log.legs if leg.covers_ground]) == 2

    def test_the_cruise_altitude_is_not_a_target(self):
        """A route flown below the nominal cruise altitude is legitimate.

        In automatic mode this is refused, because the cruise altitude is what
        the planner aims for and it has to clear the fields. In user-driven
        mode the altitudes come from the declarations instead.
        """
        route = [KSQL, declare(VPWDM, "climb", 1200), declare(KMRY, "cruise")]
        # 200 ft is below KMRY's 257 ft field elevation.
        log = nl.build_navlog(route, 200, conditions=CALM, planning_mode="manual")
        flown = [leg for leg in log.legs if leg.covers_ground]
        assert flown[0].exit_altitude_ft == pytest.approx(1200)
        assert flown[1].altitude_ft == pytest.approx(1200)  # held, not climbed
        with pytest.raises(nl.RouteError, match="not above"):
            nl.build_navlog(route, 200, conditions=CALM, planning_mode="auto")

    def test_a_level_leg_below_the_cruise_chart_warns_but_plans(self):
        """1200 ft is under the chart's 2000 ft floor; read it there and say so."""
        route = [KSQL, declare(VPWDM, "climb", 1200), declare(KMRY, "cruise")]
        log = nl.build_navlog(route, 1000, conditions=CALM, planning_mode="manual")
        assert any("bottom of the cruise chart" in w for w in log.warnings)

    def test_an_altitude_override_carries_forward(self):
        route = [
            KSQL,
            declare(VPWDM, "climb"),
            declare(KMRY, "cruise"),
        ]
        base = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        rows = [i for i, leg in enumerate(base.legs) if leg.covers_ground]
        edited = nl.build_navlog(
            route,
            7500,
            conditions=CALM,
            planning_mode="manual",
            overrides={rows[0]: nl.LegOverride(altitude_ft=4000)},
        )
        assert edited.legs[rows[0]].exit_altitude_ft == pytest.approx(4000)
        # The level leg after it inherits the edited altitude.
        assert edited.legs[rows[1]].entry_altitude_ft == pytest.approx(4000)
        assert "altitude_ft" in edited.legs[rows[0]].overridden


class TestClimbTemperatureAltitude:
    """Which altitude the climb reads its temperature at."""

    def test_climb_sees_the_air_it_climbs_through(self):
        """Two airmasses that agree at 7500 ft and disagree below it.

        The climb is integrated band by band, so cold air low down is air the
        aeroplane really does climb better in and the colder profile finishes
        sooner. Before the profile was marched this read the temperature once,
        at the top of the whole climb, and these two came out identical -- the
        conservative reading, but it charged the entire climb at the thinnest
        air it ever reaches. The conservatism now lives at band resolution
        instead; see `test_each_band_is_read_at_its_own_top`.
        """
        geo = nl.inverse(KSQL.position, KSBP.position)
        top = nl.TemperatureSample.observed(7500, 20.0)

        uniform = replace(
            CALM, temperatures=nl.TemperatureProfile.from_observations([top])
        )
        cold_below = replace(
            CALM,
            temperatures=nl.TemperatureProfile.from_observations(
                [nl.TemperatureSample.observed(0, -5.0), top]
            ),
        )
        assert cold_below.oat_c(3750) < uniform.oat_c(3750)
        assert cold_below.oat_c(7500) == pytest.approx(uniform.oat_c(7500))

        aircraft = nl.Aircraft()
        warm = pr._solve_change(0.0, 7500.0, geo, aircraft, uniform)
        cold = pr._solve_change(0.0, 7500.0, geo, aircraft, cold_below)
        assert cold["climb"].time_min < warm["climb"].time_min
        assert cold["climb"].fuel_gal < warm["climb"].fuel_gal

    def test_each_band_is_read_at_its_own_midpoint(self):
        """Within one band the temperature comes from the band's midpoint.

        The POH note is a correction to a whole climb segment, so it has to be
        read at the altitude that represents the segment -- the same midpoint
        `perf.climb_from_to` compares against ISA. Reading the band's *top*
        instead pairs a cold top-of-band temperature with a warmer mid-of-band
        standard, which on an ordinary standard day hands every band a
        deviation it does not have; now that the note is applied in both
        directions that would silently shorten every climb.

        Both profiles agree at the midpoint of the 2000-3000 ft band and one is
        colder above it, so the answers must match.
        """
        geo = nl.inverse(KSQL.position, KSBP.position)
        band_mid = nl.TemperatureSample.observed(2500, 20.0)

        uniform = replace(
            CALM, temperatures=nl.TemperatureProfile.from_observations([band_mid])
        )
        cold_above = replace(
            CALM,
            temperatures=nl.TemperatureProfile.from_observations(
                [band_mid, nl.TemperatureSample.observed(3000, 0.0)]
            ),
        )
        assert cold_above.oat_c(2900) < uniform.oat_c(2900)
        assert cold_above.oat_c(2500) == pytest.approx(uniform.oat_c(2500))

        aircraft = nl.Aircraft()
        warm = pr._solve_change(2000.0, 3000.0, geo, aircraft, uniform)
        cold = pr._solve_change(2000.0, 3000.0, geo, aircraft, cold_above)
        assert cold["climb"].time_min == pytest.approx(warm["climb"].time_min)
        assert cold["climb"].fuel_gal == pytest.approx(warm["climb"].fuel_gal)


class TestMarchedProfile:
    """Altitude changes are integrated band by band, not sampled once."""

    def test_bands_align_on_the_charts_own_rows(self):
        """Aligned on round altitudes rather than split into equal pieces.

        Two climbs that cross the same slab of air have to agree about what
        that part cost, whatever altitude each of them started from.
        """
        assert pr._altitude_bands(4.0, 3000.0) == [
            (4.0, 1000.0),
            (1000.0, 2000.0),
            (2000.0, 3000.0),
        ]
        # A change inside one band is one band, not a degenerate split.
        assert pr._altitude_bands(2000.0, 2400.0) == [(2000.0, 2400.0)]
        # Descents band identically -- each piece is solved on its own and only
        # summed, so the direction of flight does not change the split.
        assert pr._altitude_bands(3000.0, 4.0) == pr._altitude_bands(4.0, 3000.0)

    def test_time_and_fuel_telescope_across_bands(self):
        """Marching must not move the POH's own cumulative columns.

        `cum_time` and `cum_fuel` are already integrated, so differencing them
        band by band and summing has to give back exactly what differencing the
        whole climb gives. If this drifts, the bands are being read wrong.
        """
        geo = nl.inverse(KSQL.position, KSBP.position)
        marched = pr._solve_change(0.0, 9000.0, geo, nl.Aircraft(), CALM)
        whole = nl.perf.climb_from_to(
            CALM.pressure_altitude_ft(0.0),
            CALM.pressure_altitude_ft(9000.0),
            oat_c=CALM.oat_c(4500.0),
        )
        assert marched["climb"].time_min == pytest.approx(whole.time_min)
        assert marched["climb"].fuel_gal == pytest.approx(whole.fuel_gal)

    def test_wind_shear_moves_the_climb_distance(self):
        """The point of marching: a single midpoint wind sample cannot see shear.

        Both forecasts average the same wind over the climb and give the same
        wind at the midpoint, so a midpoint sample calls them identical. They
        are not -- the aeroplane spends its slow, low minutes in different air,
        and the top of climb lands in a different place.
        """
        geo = nl.inverse(KSQL.position, KSBP.position)
        course = geo.true_course_deg
        # Layered symmetrically about the climb's midpoint, so the two really
        # are indistinguishable to a single sample taken there.
        low, high = 4.0, 9500.0
        rising = replace(
            CALM,
            winds=nl.WindsAloft(
                ((low, nl.Wind(course, 5.0)), (high, nl.Wind(course, 45.0)))
            ),
        )
        falling = replace(
            CALM,
            winds=nl.WindsAloft(
                ((low, nl.Wind(course, 45.0)), (high, nl.Wind(course, 5.0)))
            ),
        )
        mid = 0.5 * (low + high)
        assert rising.winds.at(mid).speed_kt == pytest.approx(
            falling.winds.at(mid).speed_kt
        )

        aircraft = nl.Aircraft()
        up = pr._solve_change(low, high, geo, aircraft, rising)
        down = pr._solve_change(low, high, geo, aircraft, falling)
        # Climbing into a strengthening headwind covers less ground than
        # climbing out of one, by well over a mile.
        assert down["distance_nm"] - up["distance_nm"] > 1.0

    def test_reported_airspeed_reproduces_the_marched_time(self):
        """`navlog` re-flies the row; it has to land back on the integration.

        The row carries one airspeed and one altitude, and `navlog` recomputes
        ETE from them and the leg distance. That round trip has to recover the
        time the march accumulated, or the climb's fuel -- apportioned on
        `ete / climb.time_min` -- would be charged against a time nothing
        actually computed.
        """
        geo = nl.inverse(KSQL.position, KSBP.position)
        course = geo.true_course_deg
        sheared = replace(
            CALM,
            winds=nl.WindsAloft(
                ((3000.0, nl.Wind(course, 5.0)), (9000.0, nl.Wind(course, 45.0)))
            ),
        )
        change = pr._solve_change(4.0, 9500.0, geo, nl.Aircraft(), sheared)

        wind = sheared.winds.at(change["altitude_ft"])
        ground_speed = nl.solve_wind_triangle(
            course, change["tas_kt"], wind.from_deg, wind.speed_kt
        ).ground_speed_kt
        ete = 60.0 * change["distance_nm"] / ground_speed
        assert ete == pytest.approx(change["climb"].time_min)


class TestRowWindReachesTheProfile:
    """A wind typed against a row is flown by the profile, not just the row.

    It is the leg's wind, not an altitude's: it holds all the way up the climb
    and along the whole ground the row covers, which is why it replaces the
    route's wind profile outright rather than being hung at one altitude the
    way a temperature is.
    """

    @staticmethod
    def _wind(from_deg, speed_kt):
        return {1: nl.LegOverride(wind_from_deg=from_deg, wind_speed_kt=speed_kt)}

    @staticmethod
    def _climb(log):
        return next(leg for leg in log.legs if leg.phase == "climb")

    def _reaching(self, overrides):
        """A 'climb' with no altitude: the top *is* what the leg reaches."""
        route = [KSQL, declare(VPWDM, "climb"), declare(KMRY, "descent")]
        return self._climb(
            nl.build_navlog(route, 12000, conditions=CALM, overrides=overrides)
        )

    def test_a_headwind_reaches_higher_over_a_fixed_leg(self):
        """The ground distance is set by the waypoints, so wind buys minutes.

        Slower over the ground means longer in the climb, which means higher by
        the time the leg ends -- the opposite of the automatic case below,
        where the altitude is fixed and it is the distance that gives.
        """
        calm = self._reaching({})
        headwind = self._reaching(self._wind(150, 30))
        tailwind = self._reaching(self._wind(330, 30))
        assert headwind.exit_altitude_ft > calm.exit_altitude_ft
        assert tailwind.exit_altitude_ft < calm.exit_altitude_ft

    def test_the_top_of_climb_moves_in_automatic_mode(self):
        """The altitude is the target, so the headwind moves the TOC back.

        The climb takes the minutes the POH says either way; into wind it
        covers less ground in them, so it tops out sooner along the route.
        """

        def toc_at(overrides):
            log = nl.build_navlog(
                [KSQL, KSBP], 9500, conditions=CALM,
                overrides=overrides, planning_mode="auto",
            )
            return self._climb(log)

        calm = toc_at({})
        headwind = toc_at(self._wind(160, 30))
        tailwind = toc_at(self._wind(340, 30))
        assert headwind.distance_nm < calm.distance_nm < tailwind.distance_nm
        # The climb itself is unchanged: same air, same POH minutes and fuel.
        assert headwind.ete_min == pytest.approx(calm.ete_min, rel=0.02)
        assert headwind.exit_altitude_ft == pytest.approx(calm.exit_altitude_ft)

    def test_it_stops_at_the_leg_it_was_typed_on(self):
        """A wind is a leg's, unlike a temperature, which is the air's."""
        route = [KSQL, declare(VPWDM, "climb", 5500), declare(KMRY, "descent")]
        base = nl.build_navlog(route, 7500, conditions=CALM)
        edited = nl.build_navlog(
            route, 7500, conditions=CALM, overrides=self._wind(150, 30)
        )
        descent_before = next(leg for leg in base.legs if leg.phase == "descent")
        descent_after = next(leg for leg in edited.legs if leg.phase == "descent")
        assert descent_after.ground_speed_kt == pytest.approx(
            descent_before.ground_speed_kt
        )
        assert descent_after.ete_min == pytest.approx(descent_before.ete_min)

    def test_half_an_entry_leaves_the_other_half_alone(self):
        """Typing a speed with no direction must not invent a direction.

        The route's own profile answers for whatever was not typed, at whatever
        altitude the profile asks about.
        """
        winds = nl.WindsAloft.uniform(90.0, 10.0)
        typed = nl.TypedWind(winds, from_deg=None, speed_kt=25.0)
        assert typed.at(5000).from_deg == pytest.approx(90.0)
        assert typed.at(5000).speed_kt == pytest.approx(25.0)
        typed = nl.TypedWind(winds, from_deg=200.0, speed_kt=None)
        assert typed.at(5000).from_deg == pytest.approx(200.0)
        assert typed.at(5000).speed_kt == pytest.approx(10.0)

    def test_the_same_wind_holds_at_every_altitude_in_the_leg(self):
        """Not hung at one altitude and interpolated: it is the leg's wind."""
        typed = nl.TypedWind(nl.WindsAloft.calm(), from_deg=270.0, speed_kt=20.0)
        for altitude in (0.0, 3000.0, 9500.0, 14000.0):
            assert typed.at(altitude) == nl.Wind(270.0, 20.0)


# --- hybrid: events pin one end of a change, the wind moves the other ------


def point_along(start, end, nm):
    return pr.inverse(start.position, end.position).point_at_nm(nm)


def along(start, point):
    return pr.inverse(start.position, point.position).distance_nm


def with_events(end, start, *events):
    """`end` carrying `(kind, nm along the leg from start, target)` events."""
    return replace(
        end,
        events=tuple(
            pr.VerticalEvent(kind, point_along(start, end, nm), target)
            for kind, nm, target in events
        ),
    )


def course_wind(start, end, *, head):
    course = pr.inverse(start.position, end.position).true_course_deg
    from_deg = course if head else (course + 180.0) % 360.0
    return replace(CALM, winds=nl.WindsAloft.uniform(from_deg, 25.0))


def generated(route):
    return {w.name: w for w in route if w.generated}


class TestVerticalEvents:
    def test_hybrid_is_the_mode_and_auto_still_means_it(self):
        assert pr.normalise_planning_mode("auto") == "hybrid"
        assert pr.normalise_planning_mode("hybrid") == "hybrid"
        assert resolve([KSQL, KMRY], mode="hybrid") == resolve([KSQL, KMRY], mode="auto")

    def test_a_start_event_pins_the_bottom_of_climb(self):
        """Under a shelf: up to 2500 at once, then climb only after 25 nm."""
        end = with_events(KMRY, KSQL, ("complete", 10, 2500), ("start", 25, 6500))
        points = generated(resolve([KSQL, end]))
        assert list(points) == ["TOC", "BOC", "TOC2", "TOD"]
        assert along(KSQL, points["BOC"]) == pytest.approx(25.0, abs=0.05)
        assert points["BOC"].altitude_ft == pytest.approx(2500)
        assert along(KSQL, points["TOC2"]) > 25.0
        assert points["TOC2"].altitude_ft == pytest.approx(6500)

    def test_off_the_runway_a_level_off_is_climbed_to_at_once(self):
        end = with_events(KMRY, KSQL, ("complete", 10, 2500))
        toc = generated(resolve([KSQL, end]))["TOC"]
        assert along(KSQL, toc) < 10.0

    def test_a_complete_event_pins_the_top_of_climb(self):
        end = with_events(KMRY, KSQL, ("complete", 8, 2500), ("complete", 35, 6500))
        points = generated(resolve([KSQL, end]))
        assert along(KSQL, points["TOC2"]) == pytest.approx(35.0, abs=0.05)
        assert along(KSQL, points["BOC"]) < 35.0

    def test_a_headwind_lets_the_climb_start_later(self):
        """Fewer ground miles per minute of climb: the floating BOC moves
        toward the point it is pinned to."""
        end = with_events(KMRY, KSQL, ("complete", 8, 2500), ("complete", 35, 6500))
        head = generated(resolve([KSQL, end], conditions=course_wind(KSQL, KMRY, head=True)))
        tail = generated(resolve([KSQL, end], conditions=course_wind(KSQL, KMRY, head=False)))
        assert along(KSQL, head["BOC"]) > along(KSQL, tail["BOC"])
        assert along(KSQL, head["TOC2"]) == pytest.approx(along(KSQL, tail["TOC2"]), abs=0.05)

    def test_a_headwind_brings_a_floating_top_of_climb_closer(self):
        end = with_events(KMRY, KSQL, ("complete", 8, 2500), ("start", 20, 6500))
        head = generated(resolve([KSQL, end], conditions=course_wind(KSQL, KMRY, head=True)))
        tail = generated(resolve([KSQL, end], conditions=course_wind(KSQL, KMRY, head=False)))
        assert along(KSQL, head["TOC2"]) < along(KSQL, tail["TOC2"])
        assert along(KSQL, head["BOC"]) == pytest.approx(20.0, abs=0.05)

    def test_an_impossible_level_off_says_how_short_it_is(self):
        # High enough that no descent from it fits the rest of the leg either,
        # so the remainder has nothing to climb on to and levels at the point.
        end = with_events(KMRY, KSQL, ("complete", 3, 2000), ("complete", 8, 10500))
        warnings: list[str] = []
        route = pr.resolve_route(
            [KSQL, end], mode="hybrid", cruise_altitude_ft=7500,
            departure_elevation=5.0, destination_elevation=257.0,
            aircraft=nl.Aircraft(), conditions=CALM, names=pr.PhaseNamer(),
            warnings=warnings,
        )
        assert any("cannot climb to 10,500 ft" in w and "short" in w for w in warnings)
        # Short of it at the point: the climb stops where the book says it
        # gets to by then, pinned at the point, well below what was asked.
        top = generated(route)["TOC"]
        assert along(KSQL, top) == pytest.approx(8.0, abs=0.05)
        assert top.altitude_ft < 10500

    def test_step_climbs_on_one_leg(self):
        end = with_events(
            KMRY, KSQL, ("complete", 5, 2000), ("start", 15, 4500), ("start", 30, 6500)
        )
        names = list(generated(resolve([KSQL, end])))
        assert names == ["TOC", "BOC", "TOC2", "BOC2", "TOC3", "TOD"]

    def test_an_event_target_holds_down_route(self):
        """The last target on a leg is the plateau after it, as a crossing
        altitude is: the next leg does not climb back to the cruise altitude."""
        mid = with_events(VPWDM, KSQL, ("complete", 5, 4500))
        route = resolve([KSQL, mid, KMRY], altitude=7500)
        assert max(w.altitude_ft or 0 for w in route if w.generated) == pytest.approx(4500)

    def test_the_resolved_route_builds_one_segment_per_leg(self):
        end = with_events(KMRY, KSQL, ("complete", 10, 2500), ("start", 25, 6500))
        resolved = resolve([KSQL, end])
        built = segments(resolved)
        assert len(built) == len(resolved) - 1
        roles = [s.end_role for s in built]
        assert roles[:3] == ["TOC", "BOC", "TOC"]
        for a, b in pairwise(built):
            assert a.exit_altitude_ft == pytest.approx(b.entry_altitude_ft)

    def test_generated_points_know_their_drawn_leg(self):
        start = replace(KSQL, id="a")
        end = replace(with_events(KMRY, KSQL, ("complete", 10, 2500)), id="b")
        assert {w.segment_key for w in resolve([start, end]) if w.generated} == {"a>b"}

    def test_an_event_off_the_end_of_the_leg_is_said_out_loud(self):
        beyond = pr.inverse(KSQL.position, KMRY.position).point_at_nm(80)
        end = replace(KMRY, events=(pr.VerticalEvent("complete", beyond, 3000),))
        warnings: list[str] = []
        pr.resolve_route(
            [KSQL, end], mode="hybrid", cruise_altitude_ft=7500,
            departure_elevation=5.0, destination_elevation=257.0,
            aircraft=nl.Aircraft(), conditions=CALM, names=pr.PhaseNamer(),
            warnings=warnings,
        )
        assert any("beyond the end" in w for w in warnings)

    def test_a_start_event_on_the_runway_asks_for_a_level_off(self):
        end = with_events(KMRY, KSQL, ("start", 10, 4500))
        warnings: list[str] = []
        pr.resolve_route(
            [KSQL, end], mode="hybrid", cruise_altitude_ft=7500,
            departure_elevation=5.0, destination_elevation=257.0,
            aircraft=nl.Aircraft(), conditions=CALM, names=pr.PhaseNamer(),
            warnings=warnings,
        )
        assert any("field elevation" in w for w in warnings)

    def test_user_driven_mode_ignores_events(self):
        end = declare(with_events(KMRY, KSQL, ("start", 10, 4500)), "cruise")
        assert not any(w.generated for w in resolve([KSQL, end], mode="manual"))

    def test_events_survive_a_replan(self):
        end = with_events(KMRY, KSQL, ("complete", 10, 2500), ("start", 25, 6500))
        once = resolve([KSQL, end])
        assert [w.name for w in resolve(once)] == [w.name for w in once]


class TestBookClimbInTheProfile:
    def test_the_bands_inside_a_climb_are_not_each_rounded(self):
        """The climb is integrated in 1000 ft bands of *indicated* altitude,
        whose edges fall off the table's pressure-altitude rows under any
        altimeter setting. Rounding each band out to the rows doubled the
        climb: 27 min to 7500 ft where the book says 14."""
        for altimeter in (29.92, 30.12, 29.70):
            conditions = replace(CALM, altimeter_inhg=altimeter)
            log = nl.build_navlog([KSQL, KMRY], 7500, conditions=conditions, planning_mode="hybrid")
            climb = next(leg for leg in log.legs if leg.phase == "climb")
            book = nl.perf.climb_from_to(
                conditions.pressure_altitude_ft(climb.entry_altitude_ft),
                conditions.pressure_altitude_ft(climb.exit_altitude_ft),
            )
            assert climb.ete_min == pytest.approx(book.time_min, rel=0.02), altimeter


class TestALevelOffTheWindPushesPast:
    def test_a_tailwind_that_carries_the_climb_past_the_point_is_said_out_loud(self):
        """Start at 10 nm, be level by 20 nm. In a strong tailwind the climb
        from the first event is still going at the second's point."""
        end = with_events(KMRY, KSQL, ("complete", 5, 2500), ("start", 10, 7500), ("complete", 20, 7500))
        warnings: list[str] = []
        pr.resolve_route(
            [KSQL, end], mode="hybrid", cruise_altitude_ft=7500,
            departure_elevation=5.0, destination_elevation=257.0, aircraft=nl.Aircraft(),
            conditions=course_wind(KSQL, KMRY, head=False), names=pr.PhaseNamer(),
            warnings=warnings,
        )
        assert any("not at 7,500 ft by the point 20.0 nm" in w for w in warnings)

    def test_in_calm_air_it_is_met_and_nothing_is_said(self):
        end = with_events(KMRY, KSQL, ("complete", 5, 2500), ("start", 10, 7500), ("complete", 30, 7500))
        warnings: list[str] = []
        pr.resolve_route(
            [KSQL, end], mode="hybrid", cruise_altitude_ft=7500,
            departure_elevation=5.0, destination_elevation=257.0, aircraft=nl.Aircraft(),
            conditions=CALM, names=pr.PhaseNamer(), warnings=warnings,
        )
        assert not any("not at" in w for w in warnings)


class TestOneDescentNotTwo:
    """jcn99 to KMOD once flew descent, 0.3 nm level, descent: the backward
    pass estimated the descent from the airspeed and wind at its bottom only,
    the forward pass integrated it band by band, and the gap between the two
    became a step in the descent."""

    VPALT = nl.Waypoint("VPALT", LatLon(37.73916666, -121.59027777), "vfr_waypoint", altitude_ft=5500)
    JCN = nl.Waypoint("jcn99", LatLon(37.8121, -121.1821), "waypoint")
    KMOD = nl.Waypoint("KMOD", LatLon(37.6511, -120.9776), "airport", elevation_ft=99.2)
    SHEAR = replace(CALM, winds=nl.WindsAloft(((0.0, nl.Wind(250, 10)), (6000.0, nl.Wind(300, 35)))))

    def rows(self, route, conditions):
        log = nl.build_navlog(route, 5500, conditions=conditions, planning_mode="hybrid")
        return log, [leg for leg in log.legs if leg.covers_ground]

    def test_no_level_step_inside_the_descent(self):
        for conditions in (CALM, self.SHEAR):
            _, rows = self.rows([KSQL, self.VPALT, self.JCN, self.KMOD], conditions)
            phases = [leg.phase for leg in rows]
            first = phases.index("descent")
            assert all(p == "descent" for p in phases[first:]), phases

    def test_the_descent_arrives_at_pattern_altitude(self):
        for conditions in (CALM, self.SHEAR):
            _, rows = self.rows([KSQL, self.VPALT, self.JCN, self.KMOD], conditions)
            assert rows[-1].exit_altitude_ft == pytest.approx(1100, abs=5)

    def test_a_crossing_altitude_that_forces_a_late_descent_says_so(self):
        # 5,400 ft at 500 fpm and 90 KTAS needs 16.2 nm; the leg is 13.7.
        log, rows = self.rows([KSQL, self.VPALT, replace(self.JCN, altitude_ft=6500), self.KMOD], CALM)
        last = [leg for leg in rows if leg.from_name == "jcn99"]
        assert [leg.phase for leg in last] == ["descent"]
        assert any("the descent to 1,100 ft needs" in w for w in log.warnings)


    def test_the_descent_is_flown_at_90_ktas_and_500_fpm(self):
        """A true airspeed, not an indicated one: no faster up high. In still
        air the row's TAS is the setting and its time is height over rate."""
        _, rows = self.rows([KSQL, self.VPALT, self.JCN, self.KMOD], CALM)
        descents = [leg for leg in rows if leg.phase == "descent"]
        assert descents
        for leg in descents:
            assert leg.tas_kt == pytest.approx(90.0, abs=0.01)
        height = descents[0].entry_altitude_ft - descents[-1].exit_altitude_ft
        assert sum(leg.ete_min for leg in descents) == pytest.approx(height / 500.0, abs=0.01)


class TestWhatThePlannerAddsToALeg:
    """On its own the planner cuts a leg into at most three pieces -- a change,
    level, a change -- and never two changes the same way with level between
    them. A step like that is the pilot's to ask for, with events."""

    VPALT = nl.Waypoint("VPALT", LatLon(37.73916666, -121.59027777), "vfr_waypoint", id="alt")
    JCN = nl.Waypoint("jcn99", LatLon(37.8121, -121.1821), "waypoint", id="jcn")
    KMOD = nl.Waypoint("KMOD", LatLon(37.6511, -120.9776), "airport", elevation_ft=99.2, id="kmod")
    ROUTES = (
        [replace(KSQL, id="sql"), replace(KMRY, id="mry")],
        [replace(KSQL, id="sql"), replace(KSBP, id="sbp")],
        [replace(KSQL, id="sql"), replace(VPWDM, id="wdm"), replace(KMRY, id="mry")],
        [replace(KSQL, id="sql"), VPALT, JCN, KMOD],
    )
    WINDS = (
        CALM,
        replace(CALM, winds=nl.WindsAloft(((0.0, nl.Wind(250, 10)), (6000.0, nl.Wind(300, 35))))),
        replace(CALM, winds=nl.WindsAloft.uniform(150, 30)),
        replace(CALM, winds=nl.WindsAloft.uniform(330, 30)),
    )

    def by_leg(self, route, conditions, altitude):
        log = nl.build_navlog(route, altitude, conditions=conditions, planning_mode="hybrid")
        legs: dict[str, list[str]] = {}
        for leg in log.legs:
            if leg.covers_ground:
                legs.setdefault(leg.segment_key, []).append(leg.phase)
        return legs

    def test_at_most_three_pieces_and_no_step(self):
        for route in self.ROUTES:
            for conditions in self.WINDS:
                for altitude in (4500, 5500, 7500):
                    for key, phases in self.by_leg(route, conditions, altitude).items():
                        where = (key, altitude, phases)
                        assert len(phases) <= 3, where
                        for a, b, c in zip(phases, phases[1:], phases[2:]):
                            assert not (a == c != "cruise" and b == "cruise"), where

    def test_a_step_descent_the_pilot_asks_for_is_kept(self):
        """Up to 5500 by 12 nm, down to 3500 from 30 nm and level there, then
        the descent in: descent, level, descent, because the pilot said so."""
        start = replace(KSQL, id="sql")
        end = replace(
            KMRY, id="mry",
            events=(
                pr.VerticalEvent("complete", point_along(KSQL, KMRY, 12), 5500),
                pr.VerticalEvent("start", point_along(KSQL, KMRY, 30), 3500),
            ),
        )
        phases = self.by_leg([start, end], CALM, 5500)["sql>mry"]
        # climb out, level at 5500, the step down, level at 3500, the descent in
        assert phases == ["climb", "cruise", "descent", "cruise", "descent"]
