"""The weather list beside the navlog, and the loop that settles the two.

Two things are tested here and they are different. The **rule** -- a leg is
planned in the forecast over its midpoint, at the hour that midpoint is
passed -- is checked by putting a known headwind over one leg and a known
tailwind over the next. The **loop** is a fixed point, and what is checked
there is that it reaches one, that it notices when it has, and that it says so
when it cannot.

The forecasts are built by hand rather than fetched. `solve` takes them
already parsed, which is what makes any of this testable offline.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from engine import navlog as nl
from engine import planwx as pw
from engine.atmosphere import TemperatureSample
from engine.geo import LatLon, inverse

# Three fields west to east, so an easterly is a headwind the whole way and a
# westerly is a tailwind. KSQL to KLVK is short; KLVK to KMOD is longer.
KSQL = nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5)
KLVK = nl.Waypoint("KLVK", LatLon(37.6934, -121.8197), "airport", elevation_ft=400)
KMOD = nl.Waypoint("KMOD", LatLon(37.6258, -120.9544), "airport", elevation_ft=99)

OFF_BLOCKS = datetime(2026, 9, 3, 17, 0, tzinfo=UTC)
CONDITIONS = nl.Conditions(flight_date=date(2026, 9, 3))

HEADWIND = nl.WindsAloft.uniform(90.0, 30.0)  # easterly, against an eastbound leg
TAILWIND = nl.WindsAloft.uniform(270.0, 30.0)
GALE = nl.WindsAloft.uniform(180.0, 400.0)  # no heading holds a course in it


def hours(winds, count=6, temperatures=(), start=OFF_BLOCKS, step=timedelta(hours=1)):
    """A flat forecast: the same column at every hour of the window."""
    return tuple(
        pw.ForecastHour(start + n * step, winds, temperatures)
        for n in range(count)
    )


def leg(start, end, winds=None, **kwargs):
    """The forecast over one leg's midpoint."""
    return pw.PointForecast(
        name=f"{start.name}-{end.name}",
        position=inverse(start.position, end.position).point_at_fraction(0.5),
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
    return [row for row in navlog.legs if row.covers_ground]


def rows_of(solved, route):
    return nl.rows_by_drawn_leg(solved.navlog.legs, route)


class TestTheRule:
    """One forecast per leg, over its midpoint."""

    def test_each_leg_is_flown_in_its_own_midpoints_forecast(self):
        route = [KSQL, KLVK, KMOD]
        solved = solve(route, [leg(KSQL, KLVK, HEADWIND), leg(KLVK, KMOD, TAILWIND)])
        rows = rows_of(solved, route)
        assert all(r.wind_from_deg == pytest.approx(90.0) for r in rows[0])
        assert all(r.wind_from_deg == pytest.approx(270.0) for r in rows[1])

    def test_the_forecast_wind_is_reported_on_the_leg(self):
        solved = solve([KSQL, KMOD], [leg(KSQL, KMOD, HEADWIND)])
        entry = solved.weather[0]
        assert entry.has_forecast
        assert entry.wind.from_deg == pytest.approx(90.0)
        assert entry.wind.speed_kt == pytest.approx(30.0)

    def test_the_column_travels_whole_wind_and_temperature_together(self):
        warm = (TemperatureSample(6000.0, 20.0),)
        solved = solve([KSQL, KMOD], [leg(KSQL, KMOD, HEADWIND, temperatures=warm)])
        assert solved.weather[0].temperatures == warm

    def test_a_wind_nothing_can_hold_a_course_in_is_reported_not_planned_on(self):
        """Planning on it would make the route unbuildable and leave the pilot
        looking at an error instead of a plan. The leg falls back to calm and
        says what happened, which is strictly more than the refusal."""
        solved = solve([KSQL, KMOD], [leg(KSQL, KMOD, GALE)])
        entry = solved.weather[0]
        assert entry.unflyable
        assert not entry.has_forecast
        assert "cannot be flown" in entry.note
        assert flying(solved.navlog)  # and a plan came back
        assert all(r.wind_speed_kt == pytest.approx(0.0) for r in flying(solved.navlog))


class TestWhenALegIsMissing:
    def test_a_leg_with_no_forecast_is_flown_calm(self):
        route = [KSQL, KLVK, KMOD]
        solved = solve(route, [leg(KSQL, KLVK, HEADWIND), leg(KLVK, KMOD)])
        assert solved.weather[0].has_forecast
        assert not solved.weather[1].has_forecast
        assert "no forecast" in solved.weather[1].note
        rows = rows_of(solved, route)
        assert all(r.wind_speed_kt == pytest.approx(0.0) for r in rows[1])

    def test_no_forecast_anywhere_leaves_no_list(self):
        """And the plan is the one it would have been: calm, as always."""
        solved = solve([KSQL, KMOD], [leg(KSQL, KMOD)])
        assert solved.weather == ()
        assert all(r.wind_speed_kt == pytest.approx(0.0) for r in flying(solved.navlog))

    def test_a_gap_in_the_middle_still_leaves_one_entry_per_leg(self):
        """The list is read positionally against the route.

        A leg dropped rather than reported empty would shift every entry after
        it onto the wrong leg.
        """
        route = [KSQL, KLVK, KMOD]
        solved = solve(route, [leg(KSQL, KLVK), leg(KLVK, KMOD, HEADWIND)])
        assert [entry.leg for entry in solved.weather] == [0, 1]
        assert not solved.weather[0].has_forecast
        assert solved.weather[1].has_forecast

    def test_a_short_forecast_list_is_not_an_error(self):
        solved = solve([KSQL, KLVK, KMOD], [leg(KSQL, KLVK, HEADWIND)])
        assert len(solved.weather) == 2
        assert solved.weather[0].has_forecast
        assert not solved.weather[1].has_forecast


class TestTheLoop:
    def test_no_forecast_at_all_is_one_pass_and_no_list(self):
        """A plan with nothing fetched costs exactly what it always did."""
        solved = solve([KSQL, KMOD], [])
        assert solved.passes == 1
        assert solved.settled
        assert solved.weather == ()
        assert solved.notes == ()

    def test_a_settled_run_says_how_many_passes_it_took(self):
        solved = solve([KSQL, KMOD], [leg(KSQL, KMOD, HEADWIND)])
        assert solved.settled
        assert 1 < solved.passes <= pw.MAX_PASSES
        assert solved.notes == ()

    def test_a_leg_is_read_at_the_hour_its_midpoint_is_passed(self):
        """Not at departure and not at arrival: half way between the two.

        Forecast every ten minutes so the hours are fine enough to tell.
        """
        solved = solve(
            [KSQL, KMOD],
            [leg(KSQL, KMOD, HEADWIND, count=18, step=timedelta(minutes=10))],
        )
        arrival = OFF_BLOCKS + timedelta(
            minutes=next(r for r in solved.navlog.legs if r.to_name == "KMOD").cumulative_ete_min
        )
        middle = OFF_BLOCKS + (arrival - OFF_BLOCKS) / 2
        read = solved.weather[0].valid_time
        assert OFF_BLOCKS < read < arrival
        assert abs(read - middle) <= timedelta(minutes=5)

    def test_a_later_leg_is_read_at_a_later_forecast_hour(self):
        """The point of the loop: when a leg is flown decides its weather."""
        slow = nl.WindsAloft.uniform(90.0, 40.0)  # a headwind, to stretch the day out
        solved = solve(
            [KSQL, KLVK, KMOD],
            [
                leg(KSQL, KLVK, slow, count=18, step=timedelta(minutes=10)),
                leg(KLVK, KMOD, slow, count=18, step=timedelta(minutes=10)),
            ],
        )
        first, second = solved.weather
        assert second.valid_time > first.valid_time

    def test_with_no_off_blocks_every_leg_reads_the_hour_it_was_fetched_for(self):
        """Nothing about the times can move the hours, so it settles at once."""
        solved = solve(
            [KSQL, KLVK, KMOD],
            [leg(KSQL, KLVK, HEADWIND), leg(KLVK, KMOD, HEADWIND)],
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
        solved = solve([KSQL, KMOD], [leg(KSQL, KMOD, HEADWIND)], max_passes=1)
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
        blown = solve([KSQL, KMOD], [leg(KSQL, KMOD, HEADWIND)])
        calm_toc = next(r for r in flying(calm.navlog) if r.end_role == "TOC")
        blown_toc = next(r for r in flying(blown.navlog) if r.end_role == "TOC")
        assert blown_toc.distance_nm < calm_toc.distance_nm


class TestTheTwoListsAgree:
    """The navlog and the weather list describe the same flight."""

    def test_every_drawn_leg_has_exactly_one_entry_in_route_order(self):
        solved = solve(
            [KSQL, KLVK, KMOD],
            [leg(KSQL, KLVK, HEADWIND), leg(KLVK, KMOD, TAILWIND)],
        )
        assert [(e.from_name, e.to_name) for e in solved.weather] == [
            ("KSQL", "KLVK"),
            ("KLVK", "KMOD"),
        ]

    def test_each_rows_wind_is_the_one_its_leg_was_planned_in(self):
        """Including the rows the planner invented, which have no leg of their
        own: a top of climb splits a row in two without splitting the sky."""
        route = [KSQL, KLVK, KMOD]
        solved = solve(route, [leg(KSQL, KLVK, HEADWIND), leg(KLVK, KMOD, TAILWIND)])
        rows = rows_of(solved, route)
        for entry in solved.weather:
            for row in rows.get(entry.leg, []):
                expected = entry.winds.at(row.altitude_ft)
                assert row.wind_speed_kt == pytest.approx(expected.speed_kt, abs=0.1)
                assert row.wind_from_deg == pytest.approx(expected.from_deg, abs=0.1)

    def test_a_route_with_a_stop_gets_a_column_on_each_of_its_flights(self):
        """Two flights, two legs, and the weather list spans both."""
        stop = nl.Waypoint(
            "KLVK", KLVK.position, "airport", elevation_ft=400, is_landing=True
        )
        solved = solve(
            [KSQL, stop, KMOD],
            [leg(KSQL, stop, HEADWIND), leg(stop, KMOD, TAILWIND)],
        )
        assert len(solved.weather) == 2
        assert all(entry.has_forecast for entry in solved.weather)
        assert all(r.wind_speed_kt > 0 for r in flying(solved.navlog))
