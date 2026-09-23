"""Covering a route with search circles, and the endpoint that uses them.

SkyLink is asked by identifier, and the aerodromes along a route are found by
a chain of circles laid along it. What is tested here is that chain --
specifically that it has no gaps, which is the failure that would leave an
aerodrome unasked without anyone noticing -- and that the endpoint puts a real
plan through the filter.

Nothing here reaches the network. `fetch_route` is stubbed; the live API needs
a key and this suite runs offline like the rest.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine import notam as nt
from engine.geo import LatLon, direct, inverse
from server import main
from server import notams as ns

KSQL = {"name": "KSQL", "lat": 37.5119, "lon": -122.2495, "kind": "airport",
        "elevation_ft": 5, "ident": "KSQL"}
KMRY = {"name": "KMRY", "lat": 36.5870, "lon": -121.8429, "kind": "airport",
        "elevation_ft": 257, "ident": "KMRY"}


class TestTheChainOfCircles:
    """Circles along the route, and the gaps there must not be between them."""

    def route(self, length_nm):
        start = LatLon(37.0, -122.0)
        return (start, direct(start, 90.0, length_nm))

    def test_a_short_route_is_one_circle_per_end(self):
        points = ns._query_points(self.route(10.0), 25.0)
        assert len(points) >= 1
        assert all(radius == 25.0 for _, radius in points)

    def test_a_long_route_is_filled_in_between_its_waypoints(self):
        points = ns._query_points(self.route(300.0), 25.0)
        assert len(points) > 10

    @pytest.mark.parametrize("length_nm", [50.0, 137.0, 400.0])
    def test_the_chain_has_no_gaps_in_the_corridor_it_promises(self, length_nm):
        """The point of the spacing rule.

        Circles of radius R every R nm cover everything within R*sqrt(3)/2 of
        the track -- the thin spot is halfway between two centres. Sampling
        the whole corridor edge proves it: every point 20 nm off track must
        fall inside some circle, or an aerodrome there would never be asked about.
        """
        corridor = 20.0
        radius = corridor / ns._CHAIN_COVERAGE
        start, end = self.route(length_nm)
        span = inverse(start, end)
        circles = ns._query_points((start, end), radius)

        # Perpendicular to the track, not on absolute bearings: 20 nm due
        # east of an eastbound leg is ahead of it, and a point past the
        # destination is not in the corridor at all.
        edges = (
            span.true_course_deg + 90.0,
            span.true_course_deg - 90.0,
        )
        for step in range(int(span.distance_nm) + 1):
            on_track = span.point_at_nm(float(step))
            for offset in (0.0, corridor):
                for side in edges:
                    edge = direct(on_track, side, offset)
                    nearest = min(
                        inverse(centre, edge).distance_nm for centre, _ in circles
                    )
                    assert nearest <= radius + 0.5, (
                        f"{offset:.0f} nm off track at {step} nm is "
                        f"{nearest:.1f} nm from the nearest {radius:.0f} nm circle"
                    )

    def test_a_route_round_the_world_is_capped_rather_than_unbounded(self):
        """A mistyped waypoint must not turn into ten thousand requests."""
        start = LatLon(37.0, -122.0)
        points = ns._query_points((start, LatLon(37.0, 100.0)), 25.0)
        assert len(points) <= ns.MAX_QUERY_POINTS

    @pytest.mark.parametrize("length_nm", [3.0, 50.0, 137.0, 400.0])
    def test_the_destination_is_always_inside_a_circle(self, length_nm):
        """It is the one aerodrome a pilot is certain to want NOTAMs for.

        Inside a circle, not necessarily the centre of one: a destination four
        miles past the last fill point is already searched by it, and a circle
        of its own would be a second request for the same ground.
        """
        start, end = self.route(length_nm)
        points = ns._query_points((start, end), 25.0)
        assert min(inverse(centre, end).distance_nm for centre, _ in points) <= 25.0

    def test_coincident_points_do_not_get_a_circle_each(self):
        """A fill point landing next to a waypoint would search it twice."""
        start = LatLon(37.0, -122.0)
        nearby = direct(start, 90.0, 0.5)
        assert len(ns._query_points((start, nearby), 25.0)) == 1


class TestTheEndpoint:
    """A real plan through the filter, with the fetch stubbed out."""

    def request(self, **fields):
        return main.PlanRequest(
            waypoints=[KSQL, KMRY], planning_mode="auto",
            off_blocks=datetime(2026, 9, 4, 17, 0, tzinfo=UTC), **fields
        )

    def stub(self, monkeypatch, notams):
        monkeypatch.setattr(ns, "credentials_configured", lambda: True)
        monkeypatch.setattr(
            ns,
            "fetch_route",
            lambda positions, **kw: ns.RouteNotams(
                notams=tuple(notams), designators=("ZOA", "KSQL", "KMRY")
            ),
        )

    def notam(self, **fields):
        base = {
            "key": "N1", "number": "09/001", "location": "KMRY",
            "text": "RWY 10R/28L CLSD",
            "position": LatLon(36.5870, -121.8429),
            "lower_ft": 0.0, "upper_ft": 5000.0,
            "effective_start": datetime(2026, 9, 4, 12, tzinfo=UTC),
            "effective_end": datetime(2026, 9, 5, 12, tzinfo=UTC),
        }
        return nt.Notam(**{**base, **fields})

    def test_credentials_are_asked_for_before_anything_is_planned(self, monkeypatch):
        monkeypatch.setattr(ns, "credentials_configured", lambda: False)
        answer = main.notams(self.request())
        assert answer["ok"] is False
        assert answer["needs_credentials"]
        assert ns.RAPIDAPI_KEY_ENV in answer["error"]

    def test_a_notam_at_the_destination_comes_through(self, monkeypatch):
        self.stub(monkeypatch, [self.notam()])
        answer = main.notams(self.request())
        assert answer["ok"]
        assert answer["relevant"] == 1
        assert answer["notams"][0]["priority"] == "critical"
        assert answer["notams"][0]["distance_nm"] == pytest.approx(0.0, abs=0.5)

    def test_one_far_off_track_does_not(self, monkeypatch):
        self.stub(monkeypatch, [self.notam(position=LatLon(38.5, -121.5))])
        answer = main.notams(self.request())
        assert answer["relevant"] == 0
        # But the count that came back is still reported: a filter throwing
        # everything away is one worth being able to see working.
        assert answer["returned"] == 1

    def test_a_high_airway_closure_over_the_route_does_not(self, monkeypatch):
        self.stub(
            monkeypatch,
            [self.notam(lower_ft=24000.0, upper_ft=35000.0, text="ATS RTE CLSD")],
        )
        assert main.notams(self.request())["relevant"] == 0

    def test_a_surface_notam_is_kept_because_the_climb_reaches_the_surface(
        self, monkeypatch
    ):
        """The window is per row, so the climb out of KSQL covers the ground
        it started on -- not just the cruise altitude."""
        self.stub(
            monkeypatch,
            [self.notam(
                position=LatLon(37.5119, -122.2495),
                lower_ft=0.0, upper_ft=300.0,
                text="CRANE 250FT AGL",
            )],
        )
        answer = main.notams(self.request())
        assert answer["relevant"] == 1
        assert answer["notams"][0]["priority"] == "information"

    def test_one_that_expired_before_departure_does_not(self, monkeypatch):
        self.stub(
            monkeypatch,
            [self.notam(
                effective_start=datetime(2026, 9, 1, tzinfo=UTC),
                effective_end=datetime(2026, 9, 4, 9, tzinfo=UTC),
            )],
        )
        assert main.notams(self.request())["relevant"] == 0

    def test_a_wider_corridor_is_the_callers_to_ask_for(self, monkeypatch):
        off = self.notam(position=LatLon(37.5, -121.4))
        self.stub(monkeypatch, [off])
        assert main.notams(self.request())["relevant"] == 0
        assert main.notams(self.request(notam_corridor_nm=60.0))["relevant"] == 1

    def test_the_search_says_what_it_covered(self, monkeypatch):
        self.stub(monkeypatch, [self.notam()])
        answer = main.notams(self.request())
        assert answer["designators"]
        assert answer["complete"]
        assert answer["window"]["start"] and answer["window"]["end"]

    def test_a_partial_search_is_shown_and_labelled(self, monkeypatch):
        monkeypatch.setattr(ns, "credentials_configured", lambda: True)
        monkeypatch.setattr(
            ns,
            "fetch_route",
            lambda positions, **kw: ns.RouteNotams(
                notams=(self.notam(),),
                designators=("ZOA", "KSQL", "KMRY"),
                failed=("KSFO: could not reach SkyLink: timed out",),
            ),
        )
        answer = main.notams(self.request())
        assert answer["ok"]
        assert not answer["complete"]
        assert answer["failed"]


class TestTheWindow:
    """What the endpoint hands the filter, read off a real plan."""

    def window(self, off_blocks=datetime(2026, 9, 4, 17, 0, tzinfo=UTC)):
        solved, _, _ = main._build_from_request(
            main.PlanRequest(waypoints=[KSQL, KMRY], planning_mode="auto",
                             off_blocks=off_blocks)
        )
        return main._route_window(solved.navlog, off_blocks)

    def test_there_is_one_entry_per_row_that_covers_ground(self):
        window = self.window()
        assert window
        assert all(entry.span.distance_nm > 0 for entry in window)

    def test_a_climb_covers_the_band_it_climbs_through(self):
        """Not its midpoint altitude: a climb out of a field is over the field
        at the surface, and a surface NOTAM there is about it."""
        window = self.window()
        first = window[0]
        assert first.lower_ft < first.upper_ft
        assert first.lower_ft < 1000.0

    def test_the_times_run_forward_and_start_at_off_blocks(self):
        window = self.window()
        assert window[0].start == datetime(2026, 9, 4, 17, 0, tzinfo=UTC)
        for entry in window:
            assert entry.end >= entry.start
        assert window[-1].end > window[0].start

    def test_no_off_blocks_leaves_the_times_open(self):
        """Which the filter reads as 'cannot be ruled out on time'."""
        window = self.window(off_blocks=None)
        assert all(entry.start is None and entry.end is None for entry in window)
