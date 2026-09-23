"""The weather list beside the navlog, and the loop that settles the two.

Two things are tested here and they are different. The **rule** -- a leg is
planned in whichever of its two ends costs more fuel -- is a comparison, and
it is checked by putting a known headwind at one end and a known tailwind at
the other. The **loop** is a fixed point, and what is checked there is that it
reaches one, that it notices when it has, and that it says so when it cannot.

The forecasts are built by hand rather than fetched. `solve` takes them
already parsed, which is what makes any of this testable offline.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from engine import navlog as nl
from engine import planwx as pw
from engine.atmosphere import TemperatureSample
from engine.geo import LatLon

# Three fields west to east, so an easterly is a headwind the whole way and a
# westerly is a tailwind. KSQL to KLVK is short; KLVK to KMOD is longer.
KSQL = nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5)
KLVK = nl.Waypoint("KLVK", LatLon(37.6934, -121.8197), "airport", elevation_ft=400)
KMOD = nl.Waypoint("KMOD", LatLon(37.6258, -120.9544), "airport", elevation_ft=99)

OFF_BLOCKS = datetime(2026, 9, 3, 17, 0, tzinfo=UTC)
CONDITIONS = nl.Conditions(flight_date=date(2026, 9, 3))

HEADWIND = nl.WindsAloft.uniform(90.0, 30.0)  # easterly, against an eastbound leg
TAILWIND = nl.WindsAloft.uniform(270.0, 30.0)
CALM = nl.WindsAloft.calm()


def hours(winds, count=6, temperatures=(), start=OFF_BLOCKS):
    """A flat forecast: the same column at every hour of the window."""
    return tuple(
        pw.ForecastHour(start + timedelta(hours=n), winds, temperatures)
        for n in range(count)
    )


def point(waypoint, winds=None, **kwargs):
    return pw.PointForecast(
        name=waypoint.name,
        position=waypoint.position,
        hours=() if winds is None else hours(winds, **kwargs),
    )


def solve(waypoints, forecasts, off_blocks=OFF_BLOCKS, **kwargs):
    return pw.solve(
        waypoints,
        6500,
        conditions=CONDITIONS,
        forecasts=tuple(forecasts),
        off_blocks=off_blocks,
        planning_mode="auto",
        **kwargs,
    )


def flying(navlog):
    return [leg for leg in navlog.legs if leg.covers_ground]


class TestTheRule:
    """Two ends, and the one that costs more."""

    def test_the_headwind_end_is_the_one_planned_in(self):
        solved = solve([KSQL, KMOD], [point(KSQL, HEADWIND), point(KMOD, TAILWIND)])
        entry = solved.weather[0]
        assert entry.chosen == "start"
        assert entry.start_fuel_gal > entry.end_fuel_gal
        # And the rows are actually flown in it.
        assert all(leg.wind_from_deg == pytest.approx(90.0) for leg in flying(solved.navlog))

    def test_and_it_is_the_end_that_wins_not_the_departure(self):
        """The rule is 'costs more', not 'is nearer the start'."""
        solved = solve([KSQL, KMOD], [point(KSQL, TAILWIND), point(KMOD, HEADWIND)])
        assert solved.weather[0].chosen == "end"
        assert all(leg.wind_from_deg == pytest.approx(90.0) for leg in flying(solved.navlog))

    def test_the_comparison_is_shown_in_gallons(self):
        """Both ends' costs are reported, not just the winner's.

        A pilot can see from them whether the choice was close or obvious.
        """
        solved = solve([KSQL, KMOD], [point(KSQL, HEADWIND), point(KMOD, TAILWIND)])
        entry = solved.weather[0]
        assert entry.start_fuel_gal > 0
        assert entry.end_fuel_gal > 0
        assert "gal" in entry.note

    def test_an_end_nothing_can_hold_a_course_in_is_reported_not_planned_on(self):
        """The most expensive forecast there is, and the one thing not chosen.

        Planning on it would make the route unbuildable and leave the pilot
        looking at an error instead of a plan. The other end is used and the
        leg says what happened, which is strictly more than the refusal.
        """
        gale = nl.WindsAloft.uniform(180.0, 400.0)
        solved = solve([KSQL, KMOD], [point(KSQL, gale), point(KMOD, CALM)])
        entry = solved.weather[0]
        assert entry.start_unflyable
        assert entry.chosen == "end"
        assert "cannot be flown" in entry.note
        assert flying(solved.navlog)  # and a plan came back

    def test_both_ends_unflyable_leaves_the_leg_with_no_forecast(self):
        gale = nl.WindsAloft.uniform(180.0, 400.0)
        solved = solve([KSQL, KMOD], [point(KSQL, gale), point(KMOD, gale)])
        entry = solved.weather[0]
        assert entry.chosen == ""
        assert entry.start_unflyable and entry.end_unflyable
        assert "neither end" in entry.note
        assert flying(solved.navlog)

    def test_the_column_travels_whole_wind_and_temperature_together(self):
        """Not the worst wind from one end and the worst temperature from the
        other: half of one forecast against half of another describes air that
        neither of them reported."""
        warm = (TemperatureSample(6000.0, 20.0),)
        solved = solve(
            [KSQL, KMOD],
            [point(KSQL, HEADWIND, temperatures=warm), point(KMOD, TAILWIND)],
        )
        entry = solved.weather[0]
        assert entry.chosen == "start"
        assert entry.temperatures == warm


class TestWhenAnEndIsMissing:
    def test_one_end_with_no_forecast_leaves_the_other_to_it(self):
        solved = solve([KSQL, KMOD], [point(KSQL), point(KMOD, HEADWIND)])
        entry = solved.weather[0]
        assert entry.chosen == "end"
        assert "only KMOD" in entry.note

    def test_neither_end_leaves_the_leg_with_no_forecast(self):
        """And the plan is the one it would have been: calm, as always."""
        solved = solve([KSQL, KMOD], [point(KSQL), point(KMOD)])
        assert solved.weather == ()
        assert all(leg.wind_speed_kt == pytest.approx(0.0) for leg in flying(solved.navlog))

    def test_a_gap_in_the_middle_still_leaves_one_entry_per_leg(self):
        """The list is read positionally against the route.

        A leg dropped rather than reported empty would shift every entry after
        it onto the wrong leg.
        """
        solved = solve(
            [KSQL, KLVK, KMOD], [point(KSQL, HEADWIND), point(KLVK), point(KMOD)]
        )
        assert len(solved.weather) == 2
        assert [entry.leg for entry in solved.weather] == [0, 1]
        assert solved.weather[0].chosen == "start"
        assert solved.weather[1].chosen == ""
        assert "no forecast" in solved.weather[1].note

    def test_a_short_forecast_list_is_not_an_error(self):
        solved = solve([KSQL, KLVK, KMOD], [point(KSQL, HEADWIND)])
        assert len(solved.weather) == 2
        assert solved.weather[0].chosen == "start"
        assert solved.weather[1].chosen == ""


class TestTheLoop:
    def test_no_forecast_at_all_is_one_pass_and_no_list(self):
        """A plan with nothing fetched costs exactly what it always did."""
        solved = solve([KSQL, KMOD], [])
        assert solved.passes == 1
        assert solved.settled
        assert solved.weather == ()
        assert solved.notes == ()

    def test_a_settled_run_says_how_many_passes_it_took(self):
        solved = solve([KSQL, KMOD], [point(KSQL, HEADWIND), point(KMOD, TAILWIND)])
        assert solved.settled
        assert 1 < solved.passes <= pw.MAX_PASSES
        assert solved.notes == ()

    def test_a_later_leg_is_read_at_a_later_forecast_hour(self):
        """The point of the loop: when a leg is reached decides its weather.

        The second leg of this route is reached over an hour after the first,
        so it must not be planned on the departure hour.
        """
        slow = nl.WindsAloft.uniform(90.0, 40.0)  # a headwind, to stretch the day out
        solved = solve(
            [KSQL, KLVK, KMOD],
            [point(KSQL, slow), point(KLVK, slow), point(KMOD, slow)],
        )
        first, second = solved.weather
        assert first.valid_time is not None and second.valid_time is not None
        assert second.valid_time > first.valid_time

    def test_with_no_off_blocks_every_leg_reads_the_hour_it_was_fetched_for(self):
        """Nothing about the times can move the choice, so it settles at once."""
        solved = solve(
            [KSQL, KLVK, KMOD],
            [point(KSQL, HEADWIND), point(KLVK, HEADWIND), point(KMOD, HEADWIND)],
            off_blocks=None,
        )
        assert solved.settled
        assert {entry.valid_time for entry in solved.weather} == {OFF_BLOCKS}

    def test_a_run_that_never_settles_is_reported_rather_than_hidden(self):
        """The plan still comes back -- it is a real plan on a real forecast --
        but it is labelled, because a pilot reading the wind off a row is
        entitled to know it did not hold still.

        Forced by capping the passes at one rather than by hunting for a
        forecast that genuinely oscillates: what is being tested is the cap
        and what it says, and one pass can never confirm itself.
        """
        solved = solve(
            [KSQL, KMOD],
            [point(KSQL, HEADWIND), point(KMOD, TAILWIND)],
            max_passes=1,
        )
        assert not solved.settled
        assert solved.passes == 1
        assert solved.notes and "did not settle" in solved.notes[0]
        assert flying(solved.navlog)  # a plan came back anyway

    def test_the_forecast_moves_the_tops_of_climb(self):
        """Which is why the loop exists at all: the plan is not a fixed frame.

        A headwind makes the climb cover less ground, so the top of climb
        arrives sooner, so every later row is reached at a different time.
        """
        calm = solve([KSQL, KMOD], [])
        blown = solve([KSQL, KMOD], [point(KSQL, HEADWIND), point(KMOD, HEADWIND)])
        calm_toc = next(leg for leg in flying(calm.navlog) if leg.end_role == "TOC")
        blown_toc = next(leg for leg in flying(blown.navlog) if leg.end_role == "TOC")
        assert blown_toc.distance_nm < calm_toc.distance_nm


class TestTheTwoListsAgree:
    """The navlog and the weather list describe the same flight."""

    def test_every_drawn_leg_has_exactly_one_entry_in_route_order(self):
        solved = solve(
            [KSQL, KLVK, KMOD],
            [point(KSQL, HEADWIND), point(KLVK, TAILWIND), point(KMOD, HEADWIND)],
        )
        assert [(e.from_name, e.to_name) for e in solved.weather] == [
            ("KSQL", "KLVK"),
            ("KLVK", "KMOD"),
        ]

    def test_each_rows_wind_is_the_one_its_leg_was_planned_in(self):
        """Including the rows the planner invented, which have no leg of their
        own: a top of climb splits a row in two without splitting the sky."""
        solved = solve(
            [KSQL, KLVK, KMOD],
            [point(KSQL, HEADWIND), point(KLVK, CALM), point(KMOD, CALM)],
        )
        rows = nl.rows_by_drawn_leg(solved.navlog.legs, [KSQL, KLVK, KMOD])
        for entry in solved.weather:
            for row in rows.get(entry.leg, []):
                expected = entry.winds.at(row.altitude_ft)
                assert row.wind_speed_kt == pytest.approx(expected.speed_kt, abs=0.1)

    def test_a_route_with_a_stop_gets_a_column_on_each_of_its_flights(self):
        """Two flights, two legs, and the weather list spans both."""
        stop = nl.Waypoint(
            "KLVK", KLVK.position, "airport", elevation_ft=400, is_landing=True
        )
        solved = solve(
            [KSQL, stop, KMOD],
            [point(KSQL, HEADWIND), point(stop, HEADWIND), point(KMOD, TAILWIND)],
        )
        assert len(solved.weather) == 2
        assert all(entry.has_forecast for entry in solved.weather)
        assert all(leg.wind_speed_kt > 0 for leg in flying(solved.navlog))
