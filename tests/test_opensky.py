"""The OpenSky search, with the network replaced by canned answers.

What matters here is the shape of the search rather than the transport: ten
days become five windows, flights that ended elsewhere are never asked about,
tracks are fetched newest first and only until enough have passed, and a
window that fails is reported rather than silently shortening the history.
The endpoint is checked for its refusals -- a missing field, a missing ICAO
code -- which are the answers it gives without the network at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from server import main
from server import opensky as sky

NOW = datetime(2026, 10, 30, 18, 0, tzinfo=UTC)


def flight_record(icao24, departed, *, arrival="KMRY", callsign="N12345", minutes=60):
    first = int(departed.timestamp())
    return {
        "icao24": icao24,
        "firstSeen": first,
        "estDepartureAirport": "KSQL",
        "lastSeen": first + minutes * 60,
        "estArrivalAirport": arrival,
        "callsign": callsign,
    }


def track_payload(icao24, top_ft=5500.0, fixes=8):
    path = []
    for i in range(fixes):
        altitude_m = (top_ft if 0 < i < fixes - 1 else 0.0) * 0.3048
        path.append([1000 + 60 * i, 37.5 - 0.1 * i, -122.2 + 0.05 * i, altitude_m, 180.0, False])
    return {"icao24": icao24, "startTime": 1000, "endTime": 1000 + 60 * fixes, "path": path}


class FakeOpenSky:
    """Answers `_get_json` by URL, and remembers what was asked."""

    def __init__(self, flights, tracks, *, failing_windows=()):
        self.flights = flights
        self.tracks = tracks
        self.failing_windows = failing_windows
        self.asked: list[str] = []

    def __call__(self, url, ttl_s, *, refresh=False):
        self.asked.append(url)
        if "/flights/departure" in url:
            begin = int(url.split("begin=")[1].split("&")[0])
            end = int(url.split("end=")[1].split("&")[0])
            if any(b <= begin < e for b, e in self.failing_windows):
                raise sky.FlightsUnavailable("window down")
            return [f for f in self.flights if begin <= f["firstSeen"] < end]
        if "/tracks/all" in url:
            icao24 = url.split("icao24=")[1].split("&")[0]
            if icao24 not in self.tracks:
                raise sky.FlightsUnavailable("no track")
            return self.tracks[icao24]
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv(sky.CLIENT_ID_ENV, "client")
    monkeypatch.setenv(sky.CLIENT_SECRET_ENV, "secret")


def install(monkeypatch, fake):
    monkeypatch.setattr(sky, "_get_json", fake)
    return fake


class TestTheSearch:
    def test_ten_days_are_asked_as_five_two_day_windows(self, monkeypatch, configured):
        fake = install(monkeypatch, FakeOpenSky([], {}))
        found = sky.fetch_route_flights("KSQL", "KMRY", days=10, now=NOW)
        departures = [u for u in fake.asked if "/flights/departure" in u]
        assert len(departures) == 5
        assert all("airport=KSQL" in u for u in departures)
        assert found.tracks == () and found.matched == 0 and found.complete

    def test_only_flights_to_the_destination_are_asked_for_tracks_newest_first(
        self, monkeypatch, configured
    ):
        flights = [
            flight_record("aaa", NOW - timedelta(days=1)),
            flight_record("bbb", NOW - timedelta(days=2), arrival="KPAO"),
            flight_record("ccc", NOW - timedelta(days=3), callsign="UAL123"),
            flight_record("ddd", NOW - timedelta(days=9)),
        ]
        tracks = {k: track_payload(k) for k in ("aaa", "bbb", "ccc", "ddd")}
        fake = install(monkeypatch, FakeOpenSky(flights, tracks))
        found = sky.fetch_route_flights("KSQL", "KMRY", days=10, now=NOW)
        asked = [u.split("icao24=")[1].split("&")[0] for u in fake.asked if "/tracks" in u]
        assert asked == ["aaa", "ddd"]
        assert [t.flight.icao24 for t in found.tracks] == ["aaa", "ddd"]
        assert found.matched == 3  # aaa, ccc, ddd flew the pair
        assert found.candidates == 2  # ccc is an airline callsign

    def test_the_search_stops_once_enough_have_passed(self, monkeypatch, configured):
        flights = [flight_record(f"a{i:02d}", NOW - timedelta(days=1, hours=i + 1)) for i in range(8)]
        tracks = {f["icao24"]: track_payload(f["icao24"]) for f in flights}
        fake = install(monkeypatch, FakeOpenSky(flights, tracks))
        found = sky.fetch_route_flights("KSQL", "KMRY", limit=3, now=NOW)
        assert len([u for u in fake.asked if "/tracks" in u]) == 3
        assert len(found.tracks) == 3
        assert found.unexamined == 0 and found.complete

    def test_a_flight_into_class_a_is_fetched_counted_and_left_out(self, monkeypatch, configured):
        flights = [
            flight_record("hi", NOW - timedelta(days=1, hours=1)),
            flight_record("lo", NOW - timedelta(days=1, hours=2)),
            flight_record("thin", NOW - timedelta(days=1, hours=3)),
        ]
        tracks = {
            "hi": track_payload("hi", top_ft=19_000.0),
            "lo": track_payload("lo"),
            "thin": track_payload("thin", fixes=2),
        }
        install(monkeypatch, FakeOpenSky(flights, tracks))
        found = sky.fetch_route_flights("KSQL", "KMRY", now=NOW)
        assert [t.flight.icao24 for t in found.tracks] == ["lo"]
        assert found.tracks_fetched == 3
        assert found.rejected_class_a == 1
        assert found.rejected_thin == 1

    def test_a_failed_window_is_reported_not_hidden(self, monkeypatch, configured):
        flights = [flight_record("aaa", NOW - timedelta(days=1, hours=1))]
        bad_begin = int((NOW - timedelta(days=10)).timestamp())
        bad_end = int((NOW - timedelta(days=8)).timestamp())
        install(
            monkeypatch,
            FakeOpenSky(
                flights, {"aaa": track_payload("aaa")}, failing_windows=[(bad_begin, bad_end)]
            ),
        )
        found = sky.fetch_route_flights("KSQL", "KMRY", now=NOW)
        assert len(found.tracks) == 1
        assert len(found.failed_windows) == 1
        assert not found.complete

    def test_every_window_failing_raises(self, monkeypatch, configured):
        whole = [(0, 10**10)]
        install(monkeypatch, FakeOpenSky([], {}, failing_windows=whole))
        with pytest.raises(sky.FlightsUnavailable):
            sky.fetch_route_flights("KSQL", "KMRY", now=NOW)

    def test_without_credentials_the_search_runs_anonymously(self, monkeypatch):
        monkeypatch.delenv(sky.CLIENT_ID_ENV, raising=False)
        monkeypatch.delenv(sky.CLIENT_SECRET_ENV, raising=False)
        fake = install(monkeypatch, FakeOpenSky([], {}))
        found = sky.fetch_route_flights("KSQL", "KMRY", now=NOW)
        assert found.anonymous
        assert len([u for u in fake.asked if "/flights/departure" in u]) == 5

    def test_the_window_ends_where_today_begins(self, monkeypatch, configured):
        """OpenSky has nothing for today until tonight's batch, so asking
        about it would spend credits on an empty window."""
        fake = install(monkeypatch, FakeOpenSky([], {}))
        found = sky.fetch_route_flights("KSQL", "KMRY", days=10, now=NOW)
        midnight = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        assert found.end == midnight
        assert found.begin == midnight - timedelta(days=10)
        ends = [int(u.split("end=")[1].split("&")[0]) for u in fake.asked]
        assert max(ends) == int(midnight.timestamp())


class TestTheEndpoint:
    def test_an_unknown_field_is_refused_locally(self, monkeypatch):
        answer = main.flights_on_route("KZZZ9", "KMRY")
        assert answer["ok"] is False and "KZZZ9" in answer["error"]

    def test_an_anonymous_answer_says_so(self, monkeypatch):
        monkeypatch.delenv(sky.CLIENT_ID_ENV, raising=False)
        monkeypatch.delenv(sky.CLIENT_SECRET_ENV, raising=False)
        install(monkeypatch, FakeOpenSky([], {}))
        answer = main.flights_on_route("KSQL", "KMRY")
        assert answer["ok"] and answer["anonymous"]

    def test_a_found_flight_comes_through_as_compact_points(self, monkeypatch, configured):
        flights = [flight_record("aaa", NOW - timedelta(days=1, hours=1))]
        install(monkeypatch, FakeOpenSky(flights, {"aaa": track_payload("aaa")}))
        monkeypatch.setattr(sky, "datetime", _FrozenDatetime)
        answer = main.flights_on_route("KSQL", "KMRY")
        assert answer["ok"] and answer["departure"] == "KSQL" and answer["arrival"] == "KMRY"
        (flight,) = answer["flights"]
        assert flight["callsign"] == "N12345"
        assert flight["max_altitude_ft"] == 5500
        elapsed, dist, lat, lon, alt = flight["points"][1]
        assert elapsed == 60 and dist > 0 and alt == 5500
        assert answer["matched"] == 1 and answer["complete"] and not answer["anonymous"]


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW
