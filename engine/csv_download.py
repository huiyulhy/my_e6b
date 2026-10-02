"""Export a finished navlog as CSV, in the columns the navlog shows on screen.

One row per navlog row -- taxi, each piece of each leg, the pattern -- and a
totals row, with the same columns in the same order as the table in the app,
so the file and the screen can be read against each other line for line. Two
differences, both for a file read away from the map:

* **Leg Start and Leg End** name each row by the points it runs between, TOC
  and TOD included, rather than by position. The latitude and longitude the
  screen shows are left out: a pilot reading the file navigates by the names.
* **End Alt** is the altitude the row *ends* at -- what the screen's Alt column
  shows for a flown row, and what a climb is labelled by -- spelled out so it
  cannot be read as the altitude flown throughout.

Conventions kept from the screen, which a spreadsheet user would otherwise
trip over:

* **Variation is east-positive**, as everywhere else in this program: `+12.8`
  is 12.8 degrees east, and a magnetic course is the true course *minus* it.
  Printed paper pads head the column `-E / +W` and want the number the other
  way round; ours is kept so one number means one thing wherever it is read.
* **Bearings keep three digits** (`007`) and corrections keep their sign.
* **Time Rem** counts down the minutes left after each row, to the end of
  the flight, the pattern included -- the way Rem counts down the fuel.
* **Rows that go nowhere** -- taxi and the pattern -- carry their time, fuel
  and air, and leave every navigation column empty. A heading printed for a
  leg with no length would invite somebody to fly one.
"""

from __future__ import annotations

import csv
import io
import re

from engine.atmosphere import cas_from_tas
from engine.navlog import Leg, Navlog, leg_label

__all__ = ["COLUMNS", "csv_filename", "navlog_csv"]

# The navlog table's columns, left to right, minus its Lat/Lon.
COLUMNS: tuple[str, ...] = (
    "Leg Start",
    "Leg End",
    "Phase",
    "End Alt",
    "OAT",
    "PA",
    "DA",
    "TC",
    "WCA",
    "TH",
    "Var",
    "MH",
    "Wind Dir",
    "Wind Kt",
    "CAS",
    "TAS",
    "GS",
    "Dist",
    "ETE",
    "Time Rem",
    "Fuel",
    "Rem",
)

# The columns a row that goes nowhere leaves empty.
NAVIGATION_COLUMNS: tuple[str, ...] = (
    "TC", "WCA", "TH", "Var", "MH", "Wind Dir", "Wind Kt", "CAS", "TAS", "GS", "Dist",
)


def navlog_csv(navlog: Navlog) -> str:
    """The whole log as CSV, one row per navlog row plus a totals row."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="raise")
    writer.writeheader()
    for leg in navlog.legs:
        writer.writerow(_row(leg, navlog.total_time_min))
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


def _row(leg: Leg, total_min: float) -> dict[str, str]:
    """One navlog row, as the table shows it."""
    row = dict.fromkeys(COLUMNS, "")
    ground = not leg.covers_ground

    row["Leg Start"] = leg.from_name if ground else leg_label(leg.from_name, leg.start_role)
    row["Leg End"] = "" if ground else leg_label(leg.to_name, leg.end_role)
    row["Phase"] = leg.phase
    # Where the row ends up: a climb is labelled by where it gets to, not by
    # the midpoint it is solved at. A ground row is at the field, or at the
    # pattern it flies.
    row["End Alt"] = _round(
        leg.altitude_ft if ground or leg.exit_altitude_ft is None else leg.exit_altitude_ft
    )
    row["OAT"] = _round(leg.oat_c)
    row["PA"] = _round(leg.pressure_altitude_ft)
    row["DA"] = _round(leg.density_altitude_ft)
    row["ETE"] = _round(leg.ete_min, 1)
    # Minutes left after this row, to the end of the flight -- the pattern
    # included, as the fuel column counts it. Counts down to 0.
    row["Time Rem"] = _round(max(0.0, total_min - leg.cumulative_ete_min), 1)
    row["Fuel"] = _round(leg.fuel_gal, 1)
    row["Rem"] = _round(leg.fuel_remaining_gal, 1)
    if ground:
        return row

    row["TC"] = _bearing(leg.true_course_deg)
    row["WCA"] = _signed(leg.wind_correction_angle_deg)
    row["TH"] = _bearing(leg.true_heading_deg)
    # East-positive, the same number the screen shows: subtract it from the
    # true course.
    row["Var"] = _signed(leg.variation_deg)
    row["MH"] = _bearing(leg.magnetic_heading_deg)
    row["Wind Dir"] = _bearing(leg.wind_from_deg)
    row["Wind Kt"] = _round(leg.wind_speed_kt)
    row["CAS"] = _round(_cas(leg))
    row["TAS"] = _round(leg.tas_kt)
    row["GS"] = _round(leg.ground_speed_kt)
    row["Dist"] = _round(leg.distance_nm, 1)
    return row


def _totals_row(navlog: Navlog) -> dict[str, str]:
    """Distance, time, fuel and what is left -- the table's footer."""
    row = dict.fromkeys(COLUMNS, "")
    row["Leg Start"] = "Total"
    row["Dist"] = _round(navlog.total_distance_nm, 1)
    row["ETE"] = _round(navlog.total_time_min, 1)
    row["Fuel"] = _round(navlog.total_fuel_gal, 1)
    row["Rem"] = _round(navlog.fuel_remaining_gal, 1)
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


def _safe(name: str) -> str:
    """A waypoint name reduced to something a filesystem will take."""
    return re.sub(r"[^A-Za-z0-9]+", "", name)[:12]
