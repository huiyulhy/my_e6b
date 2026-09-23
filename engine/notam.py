"""NOTAMs, and which of them are actually about this flight.

A NOTAM search returns everything for a region. Most of it is not about the
flight: a crane forty miles off track, a closed taxiway at an airport being
overflown at 6,500 ft, an airway closure between FL240 and FL350, a runway
shut next Tuesday. Printing all of it is how a briefing becomes something a
pilot skims, and skimming a briefing is how the closed runway at the
destination gets missed.

So each NOTAM is put through four tests, and the ones that survive all four
are the briefing:

1. **Corridor.** Its own circle -- the Q-line centre and radius -- has to
   reach the route corridor. Twenty miles either side of track by default,
   which is a VFR pilot's realistic diversion width.
2. **Altitude.** Its band has to overlap the band the aeroplane is actually in
   over that stretch of route. A closure from FL240 to FL350 is not about a
   Skyhawk at 6,500 ft.
3. **Time.** It has to be in force while the aeroplane is there. A NOTAM that
   ends an hour before the flight, or begins the day after, is not about it.
4. **What it is.** A closed runway at the destination and a mowing crew beside
   a taxiway are not the same news, so they are sorted rather than mixed.

Each of the first three can *reject* a NOTAM, and rejecting is what this
module is for. Nothing is rejected on a guess: where a NOTAM does not say
where it is, or how high, or when, it is kept and flagged. An unreadable NOTAM
is the pilot's to read, not this program's to drop.

Pure. The fetch is `server/notams.py`, which asks SkyLink through RapidAPI.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from engine.geo import LatLon, Segment, inverse

__all__ = [
    "DEFAULT_ALTITUDE_BUFFER_FT",
    "DEFAULT_CORRIDOR_NM",
    "Notam",
    "Priority",
    "Relevance",
    "RouteWindow",
    "UnrecognisedPayload",
    "parse_skylink",
    "parse_q_line",
    "relevant",
]

# How far either side of track counts as "on the route". Twenty miles is the
# figure an EFB uses for an en-route corridor and it is a defensible one for
# VFR: it is about ten minutes of flying, which is the distance a diversion
# actually costs, and it is wide enough that a NOTAM plotted to the nearest
# minute of arc cannot fall out of it by rounding.
DEFAULT_CORRIDOR_NM = 20.0

# How far above and below the planned altitude still counts. A thousand feet
# covers the altitude a pilot actually holds against the one they filed, and
# the climb or descent they may start early.
DEFAULT_ALTITUDE_BUFFER_FT = 1000.0

# A NOTAM with no radius on it is a point. Most aerodrome NOTAMs are: the
# runway is at the airport, not within some circle of it. Given a nonzero
# default they would all reach further than they should.
_DEFAULT_RADIUS_NM = 0.0

# The Q-line vertical limits are flight levels in hundreds of feet, and the
# top of the scale means "and everything above".
_FT_PER_FLIGHT_LEVEL = 100.0
_UNLIMITED_FL = 999

# Nearer than this and the NOTAM is on the track. A Q-line position is given
# to the minute of arc, so it is only good to about a mile in the first place
# and a tenth of a mile is well inside its own error -- reporting "0.03 nm off
# track" would be precision the source does not have.
_ON_TRACK_NM = 0.1


class Priority(StrEnum):
    """What kind of news this is, which is not the same as how near it is.

    Ordered worst first, because that is the order a briefing is read in and
    the order the list is sorted in.
    """

    CRITICAL = "critical"  # runway or airport closed, TFR, navaid or GPS out
    OPERATIONAL = "operational"  # taxiway, lighting, frequency, procedure
    INFORMATION = "information"  # cranes, obstacles, mowing, minor works


_PRIORITY_ORDER = {
    Priority.CRITICAL: 0,
    Priority.OPERATIONAL: 1,
    Priority.INFORMATION: 2,
}

# Matched against the NOTAM text, worst first: the first pattern that hits
# decides. Deliberately conservative -- anything not recognised is called
# operational rather than information, because burying a NOTAM nobody
# classified is the failure that matters.
_CRITICAL = re.compile(
    r"\b(rwy|runway)\b[^.]{0,40}\b(clsd|closed)\b"
    r"|\b(ad|aerodrome|airport)\b[^.]{0,20}\b(clsd|closed)\b"
    r"|\btfr\b|temporary flight restriction"
    r"|\bprohibited\b|\brestricted area\b"
    r"|\b(vor|ndb|dme|ils|loc|gps|gnss|waas)\b[^.]{0,40}"
    r"\b(u/?s|unserviceable|out of service|otr|unusable)\b"
    r"|\bairspace\b[^.]{0,30}\b(clsd|closed)\b",
    re.IGNORECASE,
)
_INFORMATION = re.compile(
    r"\bcrane\b|\bobst\b|\bobstacle\b|\bmowing\b|\bgrass\b"
    r"|\bbird\b|\bwildlife\b|\bkite\b|\bballoon\b|\bunmanned\b|\buas\b|\bdrone\b"
    r"|\btower\b[^.]{0,30}\blgt\b|\blight(s|ing)?\b[^.]{0,20}\bu/?s\b.{0,20}\bobst\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteWindow:
    """Where the flight goes, how high, and when it is there.

    One entry per leg: the stretch of ground it covers, the band of altitude
    it covers, and the clock time it covers it at. All three are needed to
    decide whether a NOTAM is about this flight, and all three come off the
    navlog -- which is the only thing that knows them.
    """

    span: Segment
    lower_ft: float  # MSL, the lower of the leg's two ends
    upper_ft: float  # MSL, the higher
    start: datetime | None = None
    end: datetime | None = None


@dataclass(frozen=True)
class Notam:
    """One NOTAM, decoded as far as it can be decoded.

    Every geographic and temporal field is optional, and that is the point: a
    NOTAM that does not say where it is cannot be filtered out on where it is.
    `raw` keeps the payload it came from, because the decoded fields are a
    convenience and the text is the actual notice.
    """

    key: str  # a stable identity for de-duplication across overlapping queries
    number: str = ""  # "01/234"
    location: str = ""  # the ICAO location it was issued against
    kind: str = ""  # NOTAM type: N, R, C
    text: str = ""

    position: LatLon | None = None
    radius_nm: float = _DEFAULT_RADIUS_NM
    lower_ft: float | None = None  # MSL; None means "not stated"
    upper_ft: float | None = None

    effective_start: datetime | None = None
    effective_end: datetime | None = None  # None with a permanent NOTAM
    permanent: bool = False
    estimated_end: bool = False  # the end time is an "EST", not a commitment

    raw: Any = None

    @property
    def priority(self) -> Priority:
        """What kind of news this is, read from the notice's own text."""
        if _CRITICAL.search(self.text):
            return Priority.CRITICAL
        if _INFORMATION.search(self.text):
            return Priority.INFORMATION
        return Priority.OPERATIONAL

    @property
    def is_located(self) -> bool:
        return self.position is not None

    def active_during(self, start: datetime | None, end: datetime | None) -> bool:
        """Whether this is in force at any point in a window.

        Open at both ends unless the NOTAM says otherwise: a missing start is
        "already in force" and a missing end is "until further notice". Both
        are the readings that keep a NOTAM rather than drop it.
        """
        if start is None or end is None:
            return True
        if self.effective_end is not None and self.effective_end < _utc(start):
            return False
        return not (
            self.effective_start is not None and self.effective_start > _utc(end)
        )


@dataclass(frozen=True)
class Relevance:
    """One NOTAM, and why it survived -- or how near it came to not.

    The distances are carried because "12 nm off track" and "0.3 nm off track"
    are different pieces of news about the same NOTAM, and a pilot sorting a
    briefing wants the second one first.
    """

    notam: Notam
    # Nearest approach from the NOTAM's own circle to the route centreline.
    # Zero when the circle reaches the track. `None` when it has no position.
    distance_nm: float | None
    # The leg it comes nearest, as an index into the route window list.
    nearest_leg: int | None
    # Which tests it passed on the evidence, and which it passed for want of
    # any. The second is the honest half: a NOTAM with no altitude in it is
    # not "at our altitude", it is "unstated", and that is worth showing.
    reasons: tuple[str, ...] = ()

    @property
    def priority(self) -> Priority:
        return self.notam.priority

    @property
    def sort_key(self) -> tuple:
        """Worst first, then nearest first, then by number for stability."""
        return (
            _PRIORITY_ORDER[self.priority],
            math.inf if self.distance_nm is None else self.distance_nm,
            self.notam.number,
            self.notam.key,
        )


def relevant(
    notams: list[Notam] | tuple[Notam, ...],
    window: list[RouteWindow] | tuple[RouteWindow, ...],
    *,
    corridor_nm: float = DEFAULT_CORRIDOR_NM,
    altitude_buffer_ft: float = DEFAULT_ALTITUDE_BUFFER_FT,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[Relevance, ...]:
    """The NOTAMs that are about this flight, worst first.

    `start` and `end` bound the whole flight, and are the coarse time filter.
    Each leg's own window is the fine one: a NOTAM in force for an hour this
    afternoon is only relevant if the aeroplane is over it during that hour,
    not merely somewhere on the route that day.

    A NOTAM is dropped only on evidence. No position, no altitude band or no
    times means the corresponding test cannot reject it, and it comes through
    with that said in `reasons` rather than being quietly discarded or quietly
    counted as a hit.
    """
    kept: list[Relevance] = []
    for notam in notams:
        if not notam.active_during(start, end):
            continue
        verdict = _against_route(
            notam,
            tuple(window),
            corridor_nm=corridor_nm,
            altitude_buffer_ft=altitude_buffer_ft,
        )
        if verdict is not None:
            kept.append(verdict)
    return tuple(sorted(kept, key=lambda entry: entry.sort_key))


def _against_route(
    notam: Notam,
    window: tuple[RouteWindow, ...],
    *,
    corridor_nm: float,
    altitude_buffer_ft: float,
) -> Relevance | None:
    """One NOTAM against every leg, or `None` if no leg is affected.

    Leg by leg rather than against the route as a whole, because the three
    tests have to be met by the *same* leg. A NOTAM beside the first leg, at
    the altitude of the last, during the time of neither, is about no part of
    this flight -- and testing the three separately would keep it.
    """
    if not notam.is_located:
        return Relevance(
            notam=notam,
            distance_nm=None,
            nearest_leg=None,
            reasons=("no position given, so it could not be placed -- read it",),
        )

    hit: tuple[int, float] | None = None
    reasons: list[str] = []

    for index, leg in enumerate(window):
        gap = _gap_to_leg_nm(notam, leg.span)
        if gap > corridor_nm:
            continue
        if not _altitudes_overlap(notam, leg, altitude_buffer_ft):
            continue
        if not notam.active_during(leg.start, leg.end):
            continue
        if hit is None or gap < hit[1]:
            hit = (index, gap)

    if hit is None:
        return None

    index, gap = hit
    leg = window[index]
    reasons.append(
        "on the track"
        if gap <= _ON_TRACK_NM
        else f"{gap:.1f} nm off track at its nearest"
    )
    if notam.lower_ft is None and notam.upper_ft is None:
        reasons.append("no altitude band given, so the altitude could not be ruled out")
    else:
        reasons.append(
            f"{_band(notam)} against {leg.lower_ft:.0f}-{leg.upper_ft:.0f} ft on this leg"
        )
    if notam.effective_start is None and notam.effective_end is None:
        reasons.append("no times given, so the time could not be ruled out")
    elif notam.permanent:
        reasons.append("permanent")
    elif notam.estimated_end:
        reasons.append("end time is an estimate")
    return Relevance(
        notam=notam, distance_nm=gap, nearest_leg=index, reasons=tuple(reasons)
    )


def _altitudes_overlap(
    notam: Notam, leg: RouteWindow, buffer_ft: float
) -> bool:
    """Whether the NOTAM's band reaches the band the aeroplane is in.

    An unstated limit is open: no lower limit means from the surface, no upper
    limit means to the top. Both are the readings that keep a NOTAM.
    """
    low = -math.inf if notam.lower_ft is None else notam.lower_ft
    high = math.inf if notam.upper_ft is None else notam.upper_ft
    return low <= leg.upper_ft + buffer_ft and high >= leg.lower_ft - buffer_ft


def _band(notam: Notam) -> str:
    low = "SFC" if notam.lower_ft is None else f"{notam.lower_ft:.0f} ft"
    high = "unlimited" if notam.upper_ft is None else f"{notam.upper_ft:.0f} ft"
    return f"{low}-{high}"


def _gap_to_leg_nm(notam: Notam, span: Segment) -> float:
    """How far a NOTAM's circle is from a leg, in nautical miles. Zero if it
    reaches it.

    From the circle, not from its centre: a five-mile radius eighteen miles
    off track reaches a twenty-mile corridor, and a point at the same place
    does not.
    """
    if notam.position is None:
        return math.inf
    return max(0.0, _distance_to_span_nm(span, notam.position) - notam.radius_nm)


def _distance_to_span_nm(span: Segment, point: LatLon) -> float:
    """How far a point is from a leg -- the leg, not the line through it.

    Beyond either end the nearest part of the leg is that end, so the distance
    is measured to it. Using the cross-track distance alone would put a NOTAM
    two hundred miles ahead of the destination on the track, because the great
    circle through the leg does not stop where the leg does.
    """
    along = span.along_track_nm(point)
    if along < 0.0:
        return inverse(span.start, point).distance_nm
    if along > span.distance_nm:
        return inverse(span.end, point).distance_nm
    return abs(span.cross_track_nm(point))


# --- decoding SkyLink (via RapidAPI) --------------------------------------
#
# SkyLink documents its fields as "NOTAM ID, type, location, effective time,
# expiration time, and body", plus the raw text, and nothing about geometry
# or altitude. The raw text carries a Q-line with all of that in it, so the
# raw text is decoded underneath whatever was given structurally.
#
# **Written to the published description, not to a live response** -- the
# service returns 401 without a key. The field names below are the likely
# spellings of what that description lists, and the first real response
# should be checked against them.

_SKYLINK_LISTS = ("notams", "data", "results", "items")


class UnrecognisedPayload(ValueError):
    """A response that is not any shape this decoder knows.

    Raised rather than yielding an empty list, and that is the whole reason
    it exists: a provider whose format changed would otherwise report every
    airport on the route as having no NOTAMs.
    """


def parse_skylink(payload: Any) -> list[Notam]:
    """Every NOTAM in one SkyLink response.

    An empty list is only returned for a payload that was *recognisably* a
    NOTAM list with nothing in it -- a small field often has no NOTAMs, and
    that is a real answer. Anything unrecognised raises `UnrecognisedPayload`,
    because "no NOTAMs" and "could not read the reply" must never look alike.
    """
    items = _skylink_items(payload)
    found: list[Notam] = []
    for item in items:
        notam = _parse_skylink_one(item)
        if notam is not None:
            found.append(notam)
    if items and not found:
        # Every item was unreadable. That is a format change, not a quiet day.
        raise UnrecognisedPayload(
            f"{len(items)} NOTAM record(s) came back and none could be read"
        )
    return found


def _skylink_items(payload: Any) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _SKYLINK_LISTS:
            if isinstance(payload.get(key), list):
                return payload[key]
        # A single NOTAM returned bare rather than in a list.
        if _first(payload, "raw", "body", "text"):
            return [payload]
    raise UnrecognisedPayload(
        f"expected a NOTAM list, got {type(payload).__name__}"
        + (f" with keys {sorted(payload)[:6]}" if isinstance(payload, dict) else "")
    )


def _parse_skylink_one(item: Any) -> Notam | None:
    if not isinstance(item, dict):
        return None
    raw = str(_first(item, "raw", "raw_text", "rawText", "icao_message", "text") or "")
    body = str(_first(item, "body", "message", "text", "description") or "").strip()
    if not raw and not body:
        return None

    # The Q-line first, so anything given structurally lands on top of it.
    decoded = parse_q_line(raw) or parse_q_line(body)

    number = str(_first(item, "number", "notam_number", "notamNumber") or "").strip()
    identifier = str(_first(item, "id", "notam_id", "notamId") or "").strip()
    location = str(_first(item, "location", "icao", "icaoLocation", "airport") or "")
    start = _skylink_moment(
        _first(item, "effective", "effective_time", "effectiveStart", "start",
               "valid_from")
    )
    end_value = _first(item, "expiration", "expiration_time", "effectiveEnd",
                       "end", "valid_to")
    end = _skylink_moment(end_value)
    end_text = str(end_value or "").upper()

    return Notam(
        key=identifier or f"{location.strip().upper()}/{number}" or raw[:80],
        number=number or identifier,
        location=location.strip().upper(),
        kind=str(_first(item, "type", "notam_type") or "").strip(),
        text=body or raw,
        position=decoded.get("position"),
        radius_nm=decoded.get("radius_nm", _DEFAULT_RADIUS_NM),
        lower_ft=decoded.get("lower_ft"),
        upper_ft=decoded.get("upper_ft"),
        effective_start=start or decoded.get("effective_start"),
        effective_end=end or decoded.get("effective_end"),
        permanent="PERM" in end_text or decoded.get("permanent", False),
        estimated_end="EST" in end_text or decoded.get("estimated_end", False),
        raw=item,
    )


def _skylink_moment(value: Any) -> datetime | None:
    """ISO 8601 if it is that, the NOTAM's own YYMMDDHHMM if it is that."""
    return _moment(value) or (
        _yymmddhhmm(value) if isinstance(value, str) else None
    )


def _first(record: dict, *keys: str) -> Any:
    """The first of several spellings that is present and non-empty."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


# --- the ICAO Q-line ------------------------------------------------------
#
# Every NOTAM carries one, and it holds the four things this module filters
# on. A provider that returns nothing but the raw text is therefore still a
# usable provider -- which matters, because several of them do.
#
#   Q) ZOA/QRTCA/IV/NBO/W/000/100/3730N12215W025
#      FIR  code  tfc pur sco low upp  centre     radius
#
# The B) and C) lines carry the start and end as YYMMDDHHMM.

_Q_LINE = re.compile(r"^\s*Q\)\s*(.+?)\s*$", re.MULTILINE)
_B_LINE = re.compile(r"^\s*B\)\s*(\S+)", re.MULTILINE)
_C_LINE = re.compile(r"^\s*C\)\s*(\S+)", re.MULTILINE)
# The centre and radius as they appear in the Q-line's last field: degrees and
# minutes, hemisphere, then an optional three-digit radius in nautical miles.
_Q_CENTRE = re.compile(
    r"(\d{2})(\d{2})([NS])(\d{3})(\d{2})([EW])(\d{3})?", re.IGNORECASE
)


def parse_q_line(raw: str) -> dict:
    """What the Q-line and the B/C lines of a raw NOTAM say.

    Returns only the keys it could read, so a caller can lay it under whatever
    the provider gave it structurally without a missing field overwriting a
    present one. An unreadable line yields an empty mapping rather than
    raising: the text is still the notice, and half a decode beats none.

    Public because it is what makes a thin provider usable. Anything that
    hands back the raw ICAO text can be filtered in four dimensions, even if
    it parsed nothing itself.
    """
    found: dict = {}
    if not isinstance(raw, str) or not raw.strip():
        return found

    match = _Q_LINE.search(raw)
    if match:
        fields = match.group(1).split("/")
        if len(fields) >= 8:
            lower, upper = _limit_ft(fields[5]), _limit_ft(fields[6])
            if lower is not None:
                found["lower_ft"] = lower
            # An upper of 999 decodes to None, meaning unlimited, and that is
            # a real answer rather than a missing one -- but only say so when
            # the field was actually present and readable.
            if fields[6].strip():
                found["upper_ft"] = upper
            centre = _Q_CENTRE.search(fields[7])
            if centre:
                position = _q_centre(centre)
                if position is not None:
                    found["position"] = position
                if centre.group(7) is not None:
                    found["radius_nm"] = float(centre.group(7))

    start = _B_LINE.search(raw)
    if start:
        moment = _yymmddhhmm(start.group(1))
        if moment is not None:
            found["effective_start"] = moment

    end = _C_LINE.search(raw)
    if end:
        text = end.group(1).upper()
        found["permanent"] = "PERM" in text
        found["estimated_end"] = "EST" in text
        moment = _yymmddhhmm(text)
        if moment is not None:
            found["effective_end"] = moment
    return found


def _q_centre(match: re.Match) -> LatLon | None:
    lat = int(match.group(1)) + int(match.group(2)) / 60.0
    lon = int(match.group(4)) + int(match.group(5)) / 60.0
    if match.group(3).upper() == "S":
        lat = -lat
    if match.group(6).upper() == "W":
        lon = -lon
    return _latlon(lat, lon)


def _yymmddhhmm(text: str) -> datetime | None:
    """A NOTAM B/C timestamp: `2609041200` is 2026-09-04 12:00Z.

    Two-digit years, so the century has to be assumed. NOTAMs describe the
    near future and the recent past, never 1998, so 20xx is the only reading
    that is ever right.
    """
    digits = re.match(r"(\d{10})", text.strip())
    if not digits:
        return None
    try:
        return datetime.strptime(digits.group(1), "%y%m%d%H%M").replace(tzinfo=UTC)
    except ValueError:
        return None


def _latlon(lat: float, lon: float) -> LatLon | None:
    try:
        return LatLon(lat, lon)
    except ValueError:
        # A position outside the possible range is a payload error, and a
        # NOTAM with no position is still a NOTAM worth reading.
        return None


def _limit_ft(value: Any) -> float | None:
    """A Q-line vertical limit in feet MSL, or `None` where it is not stated.

    The limits are flight levels in hundreds of feet: `000` is the surface and
    `999` is the top of the scale, which means "and everything above" rather
    than 99,900 ft. Read as `None` so it reads as unlimited.
    """
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"SFC", "GND", "SURFACE"}:
        return 0.0
    if text in {"UNL", "UNLTD", "UNLIMITED"}:
        return None
    digits = re.sub(r"^(FL|A)", "", text)
    number = _as_float(digits)
    if number is None:
        return None
    if number >= _UNLIMITED_FL:
        return None
    return number * _FT_PER_FLIGHT_LEVEL


def _moment(value: Any) -> datetime | None:
    """An ISO 8601 instant, or `None` for `PERM`, `EST` and anything unreadable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _utc(moment)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
