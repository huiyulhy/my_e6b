"""Export a finished navlog as the columns of a paper navigation log.

The commercial VFR nav log pads (Jeppesen, ASA and the rest) all print the same
column set in the same order, and a pilot who has filled one in knows where to
look on it. This writes that column order out as CSV, so the plan can be pasted
into a spreadsheet shaped like the pad and read in the cockpit without
translating anything.

It is a **work-alike, not a copy**: the same columns, order and sign
conventions, none of anybody's artwork or branding.

Three things the format needs that the rest of the engine does not, and they
are the reason this is its own module:

* **The rows are checkpoints, not legs.** The pad's diagonal cells mean the
  course, wind and time written on a row belong to the leg *arriving* at the
  checkpoint that row names. Our rows are legs, so a row is labelled by where
  its leg ends -- the same "the leg arriving here" convention the profile uses
  for `segment_type`.
* **The variation column is east-positive**, as it is everywhere else in this
  program and on the navlog on screen: `+12.8` means 12.8 degrees east, and a
  magnetic course is true course *minus* variation -- east is least. Printed
  pads head the column `-E / +W` and want the number to add instead, so the
  sign is the other way round from theirs. Ours is kept, so that one number
  means one thing wherever a pilot reads it. The wind correction column,
  headed `-L / +R`, matches ours already.
* **Some columns are deliberately empty.** Deviation and compass heading come
  off the aircraft's own compass card, which this program does not model; the
  actuals -- ATE, ATA, actual ground speed -- are filled in the air. They are
  written as empty cells rather than dropped, because a blank box on a nav log
  is an instruction to the pilot and a missing column is a format that no
  longer lines up with the pad.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import time as Time

from engine.atmosphere import cas_from_tas
from engine.navlog import Leg, Navlog, leg_label

__all__ = ["COLUMNS", "csv_filename", "navlog_csv"]

# The pad's columns, left to right, flattened out of its stacked headers --
# `Wind Dir`/`Wind Vel` sit under one "Wind" heading, `ETE`/`ETA` over
# `ATE`/`ATA`, and so on. `Phase` is ours and is deliberately last, where an
# extra column falls clear of the pad's own when the file is pasted alongside
# one.
COLUMNS: tuple[str, ...] = (
    "Check Point",
    "VOR Ident",
    "VOR Freq",
    "Course (Route)",
    "Altitude",
    "Wind Dir",
    "Wind Vel",
    "Temp",
    "CAS",
    "TAS",
    "TC",
    "WCA",
    "Var",
    "TH",
    "MH",
    "Dev",
    "CH",
    "Dist Leg",
    "Dist Rem",
    "GS Est",
    "GS Act",
    "ETE",
    "ETA",
    "ATE",
    "ATA",
    "Fuel",
    "Fuel Rem",
    "GPH",
    "Phase",
)

# Columns nothing in this program can fill. Named rather than left implicit so
# that adding, say, a compass deviation card later is a matter of deleting a
# name from this tuple and filling it in.
BLANK_COLUMNS: tuple[str, ...] = (
    "VOR Ident",  # the route is flown off the map, not off a radial
    "VOR Freq",
    "Course (Route)",  # the airway or radial, if the pilot is flying one
    "Dev",  # off the aircraft's compass card
    "CH",
    "GS Act",  # the actuals, all filled in the air
    "ATE",
    "ATA",
)


def navlog_csv(navlog: Navlog, *, time_off: str | Time | None = None) -> str:
    """The whole log as CSV, one row per checkpoint plus a totals row.

    `time_off` is the departure time, as `HH:MM` or a `datetime.time`. It fills
    the ETA column; without one those cells are left empty, since an estimated
    arrival with no departure to count from would be a guess dressed as a
    number.
    """
    departure = _parse_time_off(time_off)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="raise")
    writer.writeheader()
    for leg in navlog.legs:
        writer.writerow(_row(leg, navlog, departure))
    writer.writerow(_totals_row(navlog))
    return buffer.getvalue()


def csv_filename(navlog: Navlog) -> str:
    """`navlog-KSQL-KSBP.csv`, or `navlog.csv` if the route has no names."""
    names = [w.name for w in navlog.resolved_waypoints] or [
        leg.from_name for leg in navlog.legs
    ]
    parts = [_safe(name) for name in (names[0], names[-1])] if names else []
    parts = [p for p in parts if p]
    return "-".join(["navlog", *parts]) + ".csv"


# --- one row -------------------------------------------------------------


def _row(leg: Leg, navlog: Navlog, departure: Time | None) -> dict[str, str]:
    """One checkpoint's row.

    A row that goes nowhere -- the taxi allowance, the traffic pattern -- keeps
    its fuel and its time and nothing else. Printing a course and a heading for
    a leg with no length would invite somebody to fly one.
    """
    row = dict.fromkeys(COLUMNS, "")
    ground = not leg.covers_ground

    row["Check Point"] = (
        leg.from_name if ground else leg_label(leg.to_name, leg.end_role)
    )
    row["Phase"] = leg.phase
    # The altitude the row *ends* at, which is what the navlog's own altitude
    # column shows and what a pilot edits: a climb row is labelled by where the
    # climb gets to, not by the midpoint it is solved at.
    row["Altitude"] = _round(
        leg.altitude_ft if ground else (leg.exit_altitude_ft or leg.altitude_ft)
    )
    row["Temp"] = _round(leg.oat_c)
    row["ETE"] = _round(leg.ete_min, 1)
    row["ETA"] = _clock(departure, leg.cumulative_ete_min)
    row["Fuel"] = _round(leg.fuel_gal, 1)
    row["Fuel Rem"] = _round(leg.fuel_remaining_gal, 1)
    row["GPH"] = _round(_gph(leg), 1)
    if ground:
        return row

    row["Wind Dir"] = _bearing(leg.wind_from_deg)
    row["Wind Vel"] = _round(leg.wind_speed_kt)
    row["CAS"] = _round(_cas(leg))
    row["TAS"] = _round(leg.tas_kt)
    row["TC"] = _bearing(leg.true_course_deg)
    row["WCA"] = _signed(leg.wind_correction_angle_deg)
    row["TH"] = _bearing(leg.true_heading_deg)
    # East-positive, the same number the navlog shows: subtract it from the
    # true course. A printed pad's own column is signed the other way, so this
    # is the one place the export deliberately does not follow the pad.
    row["Var"] = _signed(leg.variation_deg)
    row["MH"] = _bearing(leg.magnetic_heading_deg)
    row["Dist Leg"] = _round(leg.distance_nm, 1)
    row["Dist Rem"] = _round(
        max(0.0, navlog.total_distance_nm - leg.cumulative_distance_nm), 1
    )
    row["GS Est"] = _round(leg.ground_speed_kt)
    return row


def _totals_row(navlog: Navlog) -> dict[str, str]:
    """The pad's `Totals »` line: distance, time and fuel, and nothing else."""
    row = dict.fromkeys(COLUMNS, "")
    row["Check Point"] = "Totals"
    row["Dist Leg"] = _round(navlog.total_distance_nm, 1)
    row["ETE"] = _round(navlog.total_time_min, 1)
    row["Fuel"] = _round(navlog.total_fuel_gal, 1)
    row["Fuel Rem"] = _round(navlog.fuel_remaining_gal, 1)
    return row


# --- formatting ----------------------------------------------------------


def _round(value: float | None, places: int = 0) -> str:
    if value is None:
        return ""
    return f"{value:.{places}f}"


def _bearing(degrees: float | None) -> str:
    """Three digits, leading zeros kept: 007, not 7.

    Rounded before the wrap, not after: 359.6 is 360 degrees, which on a
    compass is 000, and a nav log column three characters wide has no room to
    say otherwise.
    """
    if degrees is None:
        return ""
    return f"{round(degrees) % 360:03d}"


def _signed(degrees: float | None) -> str:
    """A correction, with its sign always shown -- +3.8 reads as a correction."""
    if degrees is None:
        return ""
    return f"{degrees:+.1f}"


def _cas(leg: Leg) -> float | None:
    """What the airspeed indicator reads for this row's true airspeed."""
    if leg.tas_kt is None or leg.density_altitude_ft is None:
        return None
    return cas_from_tas(leg.tas_kt, leg.density_altitude_ft)


def _gph(leg: Leg) -> float | None:
    """The rate this row burned at, which is what the pad's GPH column wants."""
    if leg.ete_min <= 0:
        return None
    return leg.fuel_gal / (leg.ete_min / 60.0)


def _clock(departure: Time | None, minutes_elapsed: float) -> str:
    """`HH:MM` this many minutes after the departure time, wrapping at midnight.

    Rounded to the nearest minute, not truncated, so that the column adds up:
    a leg of 12.99 minutes prints as 13.0 in the ETE column, and an ETA that
    threw the fraction away would land a minute before the one a pilot gets by
    adding that column down the page.

    Each ETA is taken from the exact running total rather than from the printed
    ETEs, so the rounding cannot accumulate over a long route.
    """
    if departure is None:
        return ""
    total = round(departure.hour * 60 + departure.minute + minutes_elapsed)
    total %= 24 * 60
    return f"{total // 60:02d}:{total % 60:02d}"


def _parse_time_off(time_off: str | Time | None) -> Time | None:
    if time_off is None or isinstance(time_off, Time):
        return time_off
    text = time_off.strip()
    if not text:
        return None
    # `13:45` and `1345` both, since a pilot writes the second one.
    match = re.fullmatch(r"([01]?\d|2[0-3]):?([0-5]\d)", text)
    if match is None:
        raise ValueError(
            f"time off {time_off!r} is not a 24-hour time like 13:45 or 1345"
        )
    return Time(int(match.group(1)), int(match.group(2)))


def _safe(name: str) -> str:
    """A waypoint name reduced to something a filesystem will take."""
    return re.sub(r"[^A-Za-z0-9]+", "", name)[:12]
