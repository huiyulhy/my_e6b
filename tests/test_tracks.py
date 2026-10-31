"""Decoding OpenSky's flights and tracks, and the two tests that stand for VFR.

Nothing here reaches the network: every payload is written out by hand in
the shape OpenSky documents, which is also how the decoder would be checked
against the first live reply that disagrees with it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine import tracks as tr


def flight_record(**overrides):
    base = {
        "icao24": "A1B2C3",
        "firstSeen": 1_700_000_000,
        "estDepartureAirport": "KSQL",
        "lastSeen": 1_700_003_600,
        "estArrivalAirport": "KMRY",
        "callsign": "N12345  ",
        "estDepartureAirportHorizDistance": 100,
        "estDepartureAirportVertDistance": 20,
        "estArrivalAirportHorizDistance": 200,
        "estArrivalAirportVertDistance": 30,
        "departureAirportCandidatesCount": 1,
        "arrivalAirportCandidatesCount": 1,
    }
    return {**base, **overrides}


class TestTailNumbers:
    @pytest.mark.parametrize("callsign", ["N12345", "N1234A", "N12AB", "N1", "n5678b  ", "", None])
    def test_registrations_and_blanks_are_kept(self, callsign):
        assert tr.is_tail_number(callsign)

    @pytest.mark.parametrize("callsign", ["UAL123", "SKW4567", "N0123", "N12345AB", "C-GABC"])
    def test_airline_designators_and_malformed_numbers_are_not(self, callsign):
        assert not tr.is_tail_number(callsign)


class TestParsingFlights:
    def test_a_record_is_decoded_with_its_callsign_stripped(self):
        (flight,) = tr.parse_flights([flight_record()])
        assert flight.icao24 == "a1b2c3"
        assert flight.callsign == "N12345"
        assert flight.first_seen == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
        assert flight.duration_min == pytest.approx(60.0)
        assert flight.arrival == "KMRY"

    def test_an_empty_list_is_a_quiet_day_not_an_error(self):
        assert tr.parse_flights([]) == []
        assert tr.parse_flights(None) == []

    @pytest.mark.parametrize("payload", [{"error": "x"}, ["not a record"], [{"callsign": "N1"}]])
    def test_anything_unreadable_raises_rather_than_reading_as_no_flights(self, payload):
        with pytest.raises(tr.UnrecognisedPayload):
            tr.parse_flights(payload)

    def test_a_missing_time_raises(self):
        with pytest.raises(tr.UnrecognisedPayload):
            tr.parse_flights([flight_record(firstSeen=None)])


class TestChoosingTheMostRecent:
    def test_only_flights_to_the_destination_newest_first_tail_numbers_only(self):
        flights = tr.parse_flights(
            [
                flight_record(icao24="a", firstSeen=100, lastSeen=200),
                flight_record(icao24="b", firstSeen=300, lastSeen=400, estArrivalAirport="KPAO"),
                flight_record(icao24="c", firstSeen=500, lastSeen=600, callsign="UAL1"),
                flight_record(icao24="d", firstSeen=700, lastSeen=800),
                flight_record(icao24="e", firstSeen=900, lastSeen=950, estArrivalAirport="kmry"),
            ]
        )
        chosen = tr.most_recent(flights, arrival="KMRY", limit=10)
        assert [f.icao24 for f in chosen] == ["e", "d", "a"]

    def test_the_limit_holds_and_duplicates_across_windows_collapse(self):
        same = flight_record(icao24="a", firstSeen=100, lastSeen=200)
        flights = tr.parse_flights(
            [same, same, flight_record(icao24="b", firstSeen=50, lastSeen=90)]
        )
        chosen = tr.most_recent(flights, arrival="KMRY", limit=1)
        assert [f.icao24 for f in chosen] == ["a"]


def track_payload(path):
    return {"icao24": "a1b2c3", "startTime": 0, "endTime": 1, "callsign": "N12345", "path": path}


class TestParsingTracks:
    flight = tr.parse_flights([flight_record()])[0]

    def test_distance_and_time_run_along_the_track(self):
        # One degree of latitude north, then one east along the new parallel:
        # the second leg is shorter, and the run is their sum, not the chord.
        track = tr.parse_track(
            track_payload(
                [
                    [1000, 37.0, -122.0, 0.0, 0.0, True],
                    [1060, 38.0, -122.0, 304.8, 0.0, False],
                    [1120, 38.0, -121.0, 609.6, 90.0, False],
                ]
            ),
            self.flight,
        )
        first, second, third = track.points
        assert first.distance_nm == 0.0 and first.elapsed_s == 0.0
        assert second.distance_nm == pytest.approx(60.0, abs=0.2)
        assert third.distance_nm == pytest.approx(60.0 + 60.0 * 0.788, abs=0.5)
        assert third.elapsed_s == 120.0
        assert second.altitude_ft == pytest.approx(1000.0)
        assert track.max_altitude_ft == pytest.approx(2000.0)
        assert track.duration_min == pytest.approx(2.0)

    def test_a_fix_without_a_position_is_skipped_and_one_without_altitude_is_kept(self):
        track = tr.parse_track(
            track_payload(
                [
                    [1000, 37.0, -122.0, 0.0, 0.0, True],
                    [1030, None, None, 100.0, 0.0, False],
                    [1060, 37.1, -122.0, None, 0.0, False],
                ]
            ),
            self.flight,
        )
        assert len(track.points) == 2
        assert track.points[1].altitude_ft is None
        assert track.points[1].elapsed_s == 60.0

    @pytest.mark.parametrize("payload", [None, [], {"icao24": "a"}, track_payload([[1, 2, 3]])])
    def test_anything_unreadable_raises(self, payload):
        with pytest.raises(tr.UnrecognisedPayload):
            tr.parse_track(payload, self.flight)


class TestClassA:
    flight = tr.parse_flights([flight_record()])[0]

    def level(self, altitude_ft):
        metres = altitude_ft * 0.3048
        return tr.parse_track(
            track_payload(
                [
                    [1000, 37.0, -122.0, 0.0, 0.0, True],
                    [1060, 37.5, -122.0, metres, 0.0, False],
                    [1120, 38.0, -122.0, 0.0, 0.0, True],
                ]
            ),
            self.flight,
        )

    def test_a_cruise_at_vfr_altitudes_passes(self):
        assert tr.flew_below_class_a(self.level(9_500.0))

    def test_a_flight_that_reached_class_a_does_not(self):
        assert not tr.flew_below_class_a(self.level(18_000.0))
        assert not tr.flew_below_class_a(self.level(17_600.0))

    def test_no_altitude_at_all_is_kept_not_judged(self):
        track = tr.parse_track(track_payload([[1000, 37.0, -122.0, None, 0.0, False]]), self.flight)
        assert tr.flew_below_class_a(track)
