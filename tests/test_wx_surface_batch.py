"""Asking about a whole route's winds aloft in one request.

The free Open-Meteo tier counts requests per IP address, and a deployed
instance shares its address with strangers -- so a route that asked about its
waypoints one at a time was refused with a 429, and the plan fell back to the
standard atmosphere without saying so. These cover the batching that fixed it,
and the part that is easy to get quietly wrong: a reply is positional, and a
column landing on the wrong waypoint would plan the flight on another piece of
sky.

The network is stubbed at `_get_json`, so what is asserted is how many
requests would have been made and what was asked for, not the model's answer.
"""

from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from engine.geo import LatLon
from engine.weather import WeatherUnavailable
from server import wx_surface as ws
from server.main import winds_aloft_series_batch
from tests.test_aloft import HOURS, payload

NOON = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)

# A short route: three points, far enough apart to be different columns.
ROUTE = (
    LatLon(37.4611, -122.1150),
    LatLon(37.5696, -121.8738),
    LatLon(37.6258, -120.9544),
)


def many(count: int, *, elevations=None):
    """What Open-Meteo returns for `count` coordinates: a list, in order.

    Each location carries a different elevation so that a test can tell which
    column it is holding -- the whole risk being batched away is that every
    waypoint silently gets location zero's weather.
    """
    elevations = elevations or [5.0 + 100.0 * n for n in range(count)]
    return [payload(elevation_m=metres) for metres in elevations]


@pytest.fixture
def asked(monkeypatch):
    """Stub the network; collect the URLs that would have been fetched."""
    urls: list[str] = []

    def fake(url, ttl_s, *, refresh=False):
        urls.append(url)
        # Parsed rather than string-matched: the coordinates are percent
        # encoded in the query, so a naive split on "," sees one location
        # and answers with one, which is the exact failure being guarded.
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        return many(len(query["latitude"][0].split(",")))

    monkeypatch.setattr(ws, "_get_json", fake)
    return urls


class TestOneRequestForTheWholeRoute:
    def test_every_point_is_asked_about_in_a_single_call(self, asked):
        columns = ws.fetch_aloft_series_many(
            ROUTE, NOON, hours=1, ceiling_ft=3000.0
        )
        assert len(asked) == 1, "a route must cost one request, not one per point"
        assert len(columns) == len(ROUTE)

    def test_the_coordinates_are_sent_in_order(self, asked):
        ws.fetch_aloft_series_many(ROUTE, NOON, hours=1, ceiling_ft=3000.0)
        sent = urllib.parse.parse_qs(urllib.parse.urlsplit(asked[0]).query)
        assert sent["latitude"][0] == "37.4611,37.5696,37.6258"
        assert sent["longitude"][0] == "-122.1150,-121.8738,-120.9544"

    def test_each_point_gets_its_own_column(self, monkeypatch):
        """The index has to be applied, or every waypoint gets the first one."""
        monkeypatch.setattr(
            ws, "_get_json",
            lambda url, ttl, **k: many(3, elevations=[0.0, 300.0, 600.0]),
        )
        columns = ws.fetch_aloft_series_many(
            ROUTE, NOON, hours=1, ceiling_ft=3000.0
        )
        terrain = [series[0].terrain_elevation_ft for series in columns]
        # 0 m, 300 m, 600 m in feet -- distinct, and in the order asked.
        assert terrain == pytest.approx([0.0, 984.25, 1968.5], abs=0.5)

    def test_a_point_with_no_answer_keeps_its_slot(self, monkeypatch):
        """Dropping it would shift every later forecast onto the wrong point."""
        broken = many(3)
        broken[1] = {"elevation": 5.0, "hourly": {"time": []}}
        monkeypatch.setattr(ws, "_get_json", lambda url, ttl, **k: broken)
        columns = ws.fetch_aloft_series_many(
            ROUTE, NOON, hours=1, ceiling_ft=3000.0
        )
        assert len(columns) == 3
        assert columns[1] == []
        assert columns[0] and columns[2]

    def test_no_points_is_no_request(self, asked):
        assert ws.fetch_aloft_series_many((), NOON) == []
        assert asked == []

    def test_too_many_points_is_refused_before_the_network(self, asked):
        crowd = [LatLon(37.0 + n / 100.0, -122.0) for n in range(ws.MAX_BATCH_POINTS + 1)]
        with pytest.raises(WeatherUnavailable, match="the limit is"):
            ws.fetch_aloft_series_many(crowd, NOON)
        assert asked == [], "the cap must be applied without spending a request"


class TestTheSinglePointCallStillWorks:
    """`fetch_aloft_series` now goes through the batch; it must not have moved."""

    def test_it_returns_the_window_for_one_point(self, asked):
        series = ws.fetch_aloft_series(
            ROUTE[0], NOON, hours=len(HOURS), ceiling_ft=3000.0
        )
        assert len(asked) == 1
        assert [f.valid_time.hour for f in series] == [15, 16]

    def test_a_point_the_model_cannot_answer_for_still_raises(self, monkeypatch):
        """An empty forecast would read as "calm and standard"."""
        monkeypatch.setattr(
            ws, "_get_json",
            lambda url, ttl, **k: [{"elevation": 5.0, "hourly": {"time": []}}],
        )
        with pytest.raises(WeatherUnavailable, match="no hour near"):
            ws.fetch_aloft_series(ROUTE[0], NOON, ceiling_ft=3000.0)


class TestTheBatchEndpoint:
    """`/api/wx/aloft/series/batch`, as the UI calls it.

    Called directly rather than through a test client, matching the rest of
    the server tests: the routing is FastAPI's to get right, and the part
    worth pinning here is what the handler does with the arguments.
    """

    def call(self, route=ROUTE, **extra):
        # `time` is pinned: left out, the handler asks about now, and the
        # fixture payload's hours are a year away -- which `parse_aloft`
        # rightly refuses rather than snapping to.
        return winds_aloft_series_batch(**{
            "lat": ",".join(f"{p.lat}" for p in route),
            "lon": ",".join(f"{p.lon}" for p in route),
            "time": NOON.isoformat(),
            "hours": 1,
            "ceiling_ft": 3000.0,
            **extra,
        })

    def refused(self, **extra):
        """The handler, expected to refuse; returns the HTTPException."""
        with pytest.raises(HTTPException) as raised:
            self.call(**extra)
        return raised.value

    def test_it_answers_one_entry_per_point_in_order(self, asked):
        reply = self.call()
        assert reply["ok"] is True
        assert len(reply["points"]) == len(ROUTE)
        assert all(point["ok"] for point in reply["points"])
        assert len(asked) == 1

    def test_an_unanswerable_point_is_marked_rather_than_dropped(self, monkeypatch):
        broken = many(3)
        broken[1] = {"elevation": 5.0, "hourly": {"time": []}}
        monkeypatch.setattr(ws, "_get_json", lambda url, ttl, **k: broken)
        assert [point["ok"] for point in self.call()["points"]] == [True, False, True]

    def test_mismatched_coordinates_are_a_bad_request(self, asked):
        error = self.refused(lon="-122.1")
        assert error.status_code == 400
        assert "pair up" in error.detail
        assert asked == []

    def test_a_non_numeric_coordinate_is_a_bad_request(self, asked):
        assert self.refused(lat="37.5,north").status_code == 400
        assert asked == []

    def test_more_points_than_the_cap_are_a_bad_request(self, asked):
        crowd = [LatLon(37.0 + n / 100.0, -122.0) for n in range(ws.MAX_BATCH_POINTS + 1)]
        assert self.refused(route=crowd).status_code == 400
        assert asked == []

    def test_an_impossible_position_is_a_bad_request(self, asked):
        assert self.refused(lat="91.0", lon="-122.0").status_code == 400
        assert asked == []

    def test_an_hour_window_beyond_the_cap_is_a_bad_request(self, asked):
        assert self.refused(hours=99).status_code == 400
        assert asked == []

    def test_an_outage_is_data_rather_than_a_raised_error(self, monkeypatch):
        """Deliberate: the UI renders the reason instead of catching it."""
        def refused(url, ttl, **k):
            raise WeatherUnavailable(
                "could not reach api.open-meteo.com: HTTP Error 429"
            )

        monkeypatch.setattr(ws, "_get_json", refused)
        reply = self.call()
        assert reply["ok"] is False
        assert "429" in reply["error"]
