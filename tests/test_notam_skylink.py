"""SkyLink via RapidAPI: decoding its replies, and asking it about a route.

SkyLink returns 401 without a key, so the decoder is written to its published
description rather than to a captured response. What is tested here is the
part that does not depend on the exact spelling of its fields: that a reply
this code cannot read is always an outage and never "no NOTAMs", that the
Q-line in the raw text fills in what the structured fields leave out, and that
a route is turned into the right identifiers -- centres as well as aerodromes.

Nothing here reaches the network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine import notam as nt
from engine.geo import LatLon
from server import notams as ns

RAW = (
    "09/101 NOTAMN\n"
    "Q) ZOA/QMRLC/IV/NBO/A/000/050/3736N12223W005\n"
    "A) KSFO\n"
    "B) 2609041200\n"
    "C) 2609052359\n"
    "E) RWY 10R/28L CLSD"
)


def record(**fields):
    base = {"id": "N1", "type": "N", "location": "KSFO",
            "body": "RWY 10R/28L CLSD", "raw": RAW}
    return {**base, **fields}


class TestDecoding:
    def test_a_list_of_notams_decodes(self):
        found = nt.parse_skylink([record()])
        assert len(found) == 1
        assert found[0].location == "KSFO"
        assert found[0].priority is nt.Priority.CRITICAL

    @pytest.mark.parametrize("key", ["notams", "data", "results", "items"])
    def test_the_list_may_arrive_wrapped(self, key):
        assert len(nt.parse_skylink({key: [record()]})) == 1

    def test_a_single_notam_may_arrive_bare(self):
        assert len(nt.parse_skylink(record())) == 1

    def test_the_q_line_supplies_what_the_fields_leave_out(self):
        """SkyLink lists no geometry or altitude; the raw text has both."""
        one = nt.parse_skylink([record()])[0]
        assert one.position.lat == pytest.approx(37.6)
        assert one.position.lon == pytest.approx(-122.3833, abs=1e-3)
        assert one.radius_nm == pytest.approx(5.0)
        assert one.lower_ft == pytest.approx(0.0)
        assert one.upper_ft == pytest.approx(5000.0)
        assert one.effective_start == datetime(2026, 9, 4, 12, tzinfo=UTC)
        assert one.effective_end == datetime(2026, 9, 5, 23, 59, tzinfo=UTC)

    def test_structured_times_win_over_the_q_line(self):
        one = nt.parse_skylink([record(effective="2026-09-04T15:00:00Z")])[0]
        assert one.effective_start == datetime(2026, 9, 4, 15, tzinfo=UTC)

    def test_a_permanent_notam_says_so(self):
        one = nt.parse_skylink([record(expiration="PERM", raw="E) RWY 12 PAPI U/S")])[0]
        assert one.permanent
        assert one.effective_end is None

    def test_an_empty_list_is_a_real_answer(self):
        """A small field often has no NOTAMs, and saying so is correct."""
        assert nt.parse_skylink([]) == []
        assert nt.parse_skylink({"notams": []}) == []


class TestAnUnreadableReplyIsAnOutage:
    """The reason the decoder raises at all.

    A provider whose format changed would otherwise report every airport on
    the route as having no NOTAMs, and the briefing would read as clean.
    """

    @pytest.mark.parametrize(
        "payload",
        [None, "text", 42, {"message": "Invalid API key"}, {"unexpected": {"a": 1}}],
    )
    def test_an_unknown_shape_raises(self, payload):
        with pytest.raises(nt.UnrecognisedPayload):
            nt.parse_skylink(payload)

    def test_records_that_all_fail_to_read_raise(self):
        with pytest.raises(nt.UnrecognisedPayload):
            nt.parse_skylink([{"id": "N1"}, {"id": "N2"}])

    def test_one_bad_record_among_good_ones_is_skipped(self):
        assert len(nt.parse_skylink([record(), {"id": "junk"}])) == 1


class TestCredentials:
    def test_no_key_is_not_configured(self, monkeypatch):
        for name in ns.KEY_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
        assert not ns.credentials_configured()

    def test_a_key_is_configured(self, monkeypatch):
        monkeypatch.setenv(ns.SKYLINK_KEY_ENV, "k")
        assert ns.credentials_configured()

    def test_without_a_key_the_fetch_says_what_is_missing(self, monkeypatch):
        """Never an empty list: "no key" must not read as "no NOTAMs"."""
        for name in ns.KEY_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(ns.NotamsUnavailable, match=ns.SKYLINK_KEY_ENV):
            ns.fetch_route((LatLon(37.5, -122.2), LatLon(36.6, -121.8)))


class TestAskingAboutARoute:
    """Turning a route into identifiers, and what happens when they fail."""

    ROUTE = (LatLon(37.5119, -122.2495), LatLon(36.5870, -121.8429))

    @pytest.fixture(autouse=True)
    def skylink(self, monkeypatch):
        monkeypatch.setenv(ns.SKYLINK_KEY_ENV, "k")

    def answer(self, monkeypatch, reply):
        """Stub the network: `reply(designator)` returns the payload or raises."""
        asked: list[str] = []

        def fake(url, **kwargs):
            designator = url.rsplit("/", 1)[-1]
            asked.append(designator)
            return reply(designator)

        monkeypatch.setattr(ns, "_get_json", fake)
        return asked

    def test_the_centre_is_asked_about_as_well_as_the_aerodromes(self, monkeypatch):
        """TFRs and MOAs are filed against the ARTCC, not any airport."""
        asked = self.answer(monkeypatch, lambda d: [])
        found = ns.fetch_route(self.ROUTE)
        assert "ZOA" in asked
        assert "KSQL" in asked and "KMRY" in asked
        assert found.complete

    def test_centres_are_asked_first_so_the_budget_never_cuts_them(self, monkeypatch):
        asked = self.answer(monkeypatch, lambda d: [])
        ns.fetch_route(self.ROUTE)
        assert asked[0] == "ZOA"

    def test_notams_from_several_identifiers_are_merged(self, monkeypatch):
        def reply(designator):
            if designator == "KSFO":
                return [record()]
            if designator == "ZOA":
                return [record(id="N2", location="ZOA", body="TFR VIP MOVEMENT")]
            return []

        self.answer(monkeypatch, reply)
        assert {n.key for n in ns.fetch_route(self.ROUTE).notams} == {"N1", "N2"}

    def test_an_unreadable_reply_is_a_failure_not_an_empty_airport(self, monkeypatch):
        def reply(designator):
            return {"surprise": True} if designator == "KSFO" else []

        self.answer(monkeypatch, reply)
        found = ns.fetch_route(self.ROUTE)
        assert not found.complete
        assert any("KSFO" in entry and "unreadable" in entry for entry in found.failed)

    def test_every_query_failing_raises_rather_than_returning_nothing(self, monkeypatch):
        def reply(designator):
            raise ns.NotamsUnavailable("SkyLink rejected the credentials (403)")

        self.answer(monkeypatch, reply)
        with pytest.raises(ns.NotamsUnavailable, match="no SkyLink query succeeded"):
            ns.fetch_route(self.ROUTE)

    def test_a_total_outage_still_raises_when_the_budget_also_cut_some(self, monkeypatch):
        """The bug this guards: the budget notice lands in the failure list,
        and must not be able to make a total outage look like an empty
        briefing that was merely trimmed."""
        monkeypatch.setattr(ns, "MAX_DESIGNATORS", 3)

        def reply(designator):
            raise ns.NotamsUnavailable("could not reach SkyLink: timed out")

        self.answer(monkeypatch, reply)
        with pytest.raises(ns.NotamsUnavailable):
            ns.fetch_route(self.ROUTE)

    def test_the_budget_cap_is_reported_not_silent(self, monkeypatch):
        monkeypatch.setattr(ns, "MAX_DESIGNATORS", 3)
        asked = self.answer(monkeypatch, lambda d: [])
        found = ns.fetch_route(self.ROUTE)
        assert len(asked) == 3
        assert not found.complete
        assert any("allowance" in entry for entry in found.failed)

    def test_the_route_fields_are_asked_first_so_the_cap_never_cuts_them(
        self, monkeypatch
    ):
        """The bug this guards: walking from departure to destination and
        cutting at the cap dropped the destination on every busy route.
        O'Hare to Indianapolis has ninety identifiers within reach."""
        from engine import airports as apt

        route = (apt.find("KORD").position, apt.find("KIND").position)
        asked = self.answer(monkeypatch, lambda d: [])
        found = ns.fetch_route(route, priority=("KORD", "KIND"))
        assert asked[:2] == ["KORD", "KIND"]
        assert "KIND" in found.designators
        assert not found.complete  # and the cap is still reported

    def test_the_centres_come_straight_after_the_route_fields(self, monkeypatch):
        from engine import airports as apt

        route = (apt.find("KORD").position, apt.find("KIND").position)
        asked = self.answer(monkeypatch, lambda d: [])
        ns.fetch_route(route, priority=("KORD", "KIND"))
        centres = [d for d in asked if len(d) == 3 and d.startswith("Z")]
        assert centres and asked[2 : 2 + len(centres)] == centres

    def test_airports_are_searched_for_wide_enough_to_cover_the_corridor(
        self, monkeypatch
    ):
        """Circles of radius R every R cover only R*sqrt(3)/2 of track, so
        searching at the corridor width itself missed an aerodrome 19 nm off
        track halfway between two samples. The chain test in
        `test_notam_fetch.py` proves this radius has no gaps."""
        from engine import airports as apt

        seen = {}
        real = apt.designators_along_route

        def spy(samples, radius, **kwargs):
            seen["radius"] = radius
            return real(samples, radius, **kwargs)

        monkeypatch.setattr(apt, "designators_along_route", spy)
        self.answer(monkeypatch, lambda d: [])
        ns.fetch_route(self.ROUTE, corridor_nm=20.0)
        assert seen["radius"] == pytest.approx(20.0 / ns._CHAIN_COVERAGE)


class TestTheRequest:
    """What actually goes over the wire.

    A misspelt auth header is a 401 on every request, forever, and would read
    as a bad key rather than as a bug. SkyLink sells the same service through
    two channels with different headers -- `x-api-key` on a direct licence,
    `X-RapidAPI-Key` through the marketplace -- so which one is sent, and to
    which host, is worth pinning rather than assuming.
    """

    def test_the_licence_key_is_sent_to_the_direct_endpoint(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(ns.SKYLINK_KEY_ENV, "secret")
        monkeypatch.setattr(ns, "CACHE", tmp_path)
        seen = {}

        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"[]"

        def urlopen(request, timeout):
            # `Request` normalises header names to capitalised form.
            seen["key"] = request.get_header("X-api-key")
            seen["rapidapi"] = request.get_header("X-rapidapi-key")
            seen["host"] = request.get_header("X-rapidapi-host")
            seen["url"] = request.full_url
            return Reply()

        monkeypatch.setattr(ns.urllib.request, "urlopen", urlopen)
        assert ns._get_json(f"{ns.SKYLINK_API}/KSQL", refresh=True) == []
        assert seen["key"] == "secret"
        # The marketplace headers belong to the other channel. Sending them
        # as well would not fail loudly, it would just be noise on every
        # request -- and noise that suggests the wrong endpoint to a reader.
        assert seen["rapidapi"] is None
        assert seen["host"] is None
        assert seen["url"] == "https://data.skylinkapi.com/v3/notams/KSQL"
