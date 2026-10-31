"""Flights that actually flew a city pair, and the shape each one made.

The pure half of the flown-routes feature. `server/opensky.py` fetches the
raw OpenSky payloads; everything here decodes them and decides what they
mean, offline, so it can be tested without an account and run under Pyodide.

**What "VFR" means here, and why it is a guess.** ADS-B carries no flight
rules. A transponder says where an aircraft is, not what clearance it is on,
and OpenSky's flight records say only which fields a track started and ended
near. So "VFR flights only" has to be read off two proxies, both stated here
so a reader can judge them:

- The callsign is a US registration (`N12345`, `N1234A`). An airline or
  fractional flight flies under an ICAO three-letter designator (`UAL123`,
  `SKW4567`), and those are IFR as a matter of course. A GA aeroplane flying
  under its tail number may be IFR too, which is the half this test cannot
  see -- but a tail-number callsign is at least the population a VFR pilot
  wants to learn from.
- The track never reaches Class A, which starts at 18,000 ft. A flight that
  did was on an IFR clearance by definition, and nothing a VFR pilot can
  copy. The bar is set a little under the floor because the barometric
  altitude ADS-B reports is referenced to 29.92 and quantised to 25 or 100 ft.

Both tests keep rather than drop when the evidence is missing: a flight
with no callsign is kept, since a blank callsign is common on light aircraft
and the alternative is throwing away exactly the traffic being looked for.

**Distance along the track is summed point to point,** not measured from the
departure, so a flight that wandered -- a practice area, a sightseeing loop
-- shows its true length on the profile rather than folding back on itself.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from engine.geo import LatLon, distance_nm

__all__ = [
    "CLASS_A_FLOOR_FT",
    "Flight",
    "FlightTrack",
    "TrackPoint",
    "UnrecognisedPayload",
    "flew_below_class_a",
    "is_tail_number",
    "most_recent",
    "parse_flights",
    "parse_track",
]

# Where Class A begins, and the altitude a track is tested against. The test
# itself sits a little below it (see `flew_below_class_a`) to allow for the
# quantisation of what ADS-B reports.
CLASS_A_FLOOR_FT = 18_000.0
_CLASS_A_MARGIN_FT = 500.0

_FT_PER_M = 1.0 / 0.3048

# A US registration as a callsign: N, then one to five characters in all --
# digits first, not starting with zero, then up to two letters, never O or I
# (the FAA does not issue them). Padded with spaces in the OpenSky record,
# hence the strip before the match.
_TAIL_NUMBER = re.compile(r"^N(?=[0-9A-Z]{1,5}$)[1-9]\d{0,4}[A-HJ-NP-Z]{0,2}$")

# Too few fixes to draw: a "track" of two points is a straight line between
# the places the receivers happened to hear it, not a route anyone flew.
MIN_TRACK_POINTS = 5


class UnrecognisedPayload(ValueError):
    """OpenSky answered with something this decoder does not know how to read."""


@dataclass(frozen=True)
class Flight:
    """One record from OpenSky's flights-by-airport endpoints."""

    icao24: str
    callsign: str  # stripped; "" when the record has none
    first_seen: datetime
    last_seen: datetime
    departure: str | None  # estimated, ICAO
    arrival: str | None  # estimated, ICAO

    @property
    def duration_min(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds() / 60.0

    @property
    def midpoint(self) -> datetime:
        """A moment inside the flight, which is what the tracks endpoint wants."""
        return self.first_seen + (self.last_seen - self.first_seen) / 2


@dataclass(frozen=True)
class TrackPoint:
    time: datetime
    position: LatLon
    altitude_ft: float | None  # barometric, 29.92; None when not reported
    on_ground: bool
    elapsed_s: float  # since the first point of the track
    distance_nm: float  # summed along the track from its first point


@dataclass(frozen=True)
class FlightTrack:
    flight: Flight
    points: tuple[TrackPoint, ...]

    @property
    def max_altitude_ft(self) -> float | None:
        altitudes = [p.altitude_ft for p in self.points if p.altitude_ft is not None]
        return max(altitudes) if altitudes else None

    @property
    def length_nm(self) -> float:
        return self.points[-1].distance_nm if self.points else 0.0

    @property
    def duration_min(self) -> float:
        return self.points[-1].elapsed_s / 60.0 if self.points else 0.0


def is_tail_number(callsign: str | None) -> bool:
    """Whether a callsign is a US registration rather than an airline flight.

    A blank callsign is accepted: it is the common case for a light aircraft
    with a transponder nobody configured, and it is not evidence of an
    airline.
    """
    cleaned = (callsign or "").strip().upper()
    if not cleaned:
        return True
    return bool(_TAIL_NUMBER.match(cleaned))


def flew_below_class_a(track: FlightTrack) -> bool:
    """Whether the track stayed under Class A, and so could have been VFR.

    A track with no altitude at all is kept: it says nothing either way, and
    dropping it would hide a flight for a sensor gap rather than for a reason.
    """
    top = track.max_altitude_ft
    return top is None or top < CLASS_A_FLOOR_FT - _CLASS_A_MARGIN_FT


def most_recent(
    flights: list[Flight] | tuple[Flight, ...],
    *,
    arrival: str,
    limit: int,
) -> list[Flight]:
    """The flights that ended at `arrival`, newest first, tail numbers only.

    De-duplicated by aircraft and departure time: the same flight can appear
    in two overlapping query windows.
    """
    wanted = arrival.strip().upper()
    seen: set[tuple[str, datetime]] = set()
    kept: list[Flight] = []
    for flight in sorted(flights, key=lambda f: f.first_seen, reverse=True):
        if (flight.arrival or "").upper() != wanted:
            continue
        if not is_tail_number(flight.callsign):
            continue
        key = (flight.icao24, flight.first_seen)
        if key in seen:
            continue
        seen.add(key)
        kept.append(flight)
        if len(kept) >= limit:
            break
    return kept


# --- decoding --------------------------------------------------------------


def _moment(value: Any, field: str) -> datetime:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise UnrecognisedPayload(f"flight record has no usable {field}")
    return datetime.fromtimestamp(float(value), tz=UTC)


def parse_flights(payload: Any) -> list[Flight]:
    """Decode a flights-by-airport reply.

    An empty list is a recognisable answer -- a quiet field on a quiet day.
    Anything that is not a list of records with the documented keys raises,
    because an unreadable reply shown as "no flights" is the one failure a
    reader cannot see.
    """
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise UnrecognisedPayload("expected a list of flight records")
    flights: list[Flight] = []
    for record in payload:
        if not isinstance(record, dict) or "icao24" not in record:
            raise UnrecognisedPayload("flight record is not a dictionary with an icao24")
        flights.append(
            Flight(
                icao24=str(record["icao24"]).lower(),
                callsign=str(record.get("callsign") or "").strip(),
                first_seen=_moment(record.get("firstSeen"), "firstSeen"),
                last_seen=_moment(record.get("lastSeen"), "lastSeen"),
                departure=record.get("estDepartureAirport"),
                arrival=record.get("estArrivalAirport"),
            )
        )
    return flights


def parse_track(payload: Any, flight: Flight) -> FlightTrack:
    """Decode a tracks reply into points with distance and time along them.

    OpenSky's path entries are positional: `[time, lat, lon, baro_altitude_m,
    true_track, on_ground]`. A fix with no position is skipped; one with no
    altitude is kept with `altitude_ft` None, since the ground track is still
    worth drawing.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("path"), list):
        raise UnrecognisedPayload("expected a track with a path")

    points: list[TrackPoint] = []
    previous: LatLon | None = None
    start: datetime | None = None
    run_nm = 0.0
    for fix in payload["path"]:
        if not isinstance(fix, list | tuple) or len(fix) < 6:
            raise UnrecognisedPayload("track fix is not the documented six fields")
        time, lat, lon, baro_m, _true_track, on_ground = fix[:6]
        if lat is None or lon is None:
            continue
        position = LatLon(float(lat), float(lon))
        moment = _moment(time, "time")
        if start is None:
            start = moment
        if previous is not None:
            run_nm += distance_nm(previous, position)
        altitude = None
        if baro_m is not None and math.isfinite(float(baro_m)):
            altitude = float(baro_m) * _FT_PER_M
        points.append(
            TrackPoint(
                time=moment,
                position=position,
                altitude_ft=altitude,
                on_ground=bool(on_ground),
                elapsed_s=(moment - start).total_seconds(),
                distance_nm=run_nm,
            )
        )
        previous = position
    return FlightTrack(flight=flight, points=tuple(points))
