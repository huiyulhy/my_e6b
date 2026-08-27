"""How old the bundled data is, and when it stops being usable.

An offline planner has no way to notice that its data went stale -- there is
no server to tell it. So the two datasets that expire on a clock carry their
own dates, and the UI shows them:

* FAA NASR, reissued every 28 days. Treated as good for one month from the
  effective date printed in `data/nasr/README.txt`.
* NOAA World Magnetic Model, reissued every five years. The window comes
  from the epoch in the coefficient file itself.

Nothing here refuses to plan on expired data -- that is the pilot's call and
the disclaimer already covers it. It only makes the age visible.
"""

from __future__ import annotations

import calendar
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from engine import magnetic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NASR_README = DATA_DIR / "nasr" / "README.txt"
AIRPORT_DB = DATA_DIR / "aero" / "airports.sqlite"

NASR_VALID_MONTHS = 1

# Check the date, in the first line of every NASR README
_EFFECTIVE_LINE = re.compile(r"effective date\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})")
_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}


@dataclass(frozen=True)
class Dataset:
    """One dated dataset and whether it is still current."""

    key: str
    label: str
    effective: date | None
    expires: date | None
    # Set when the date could not be read at all, or when what is on disk
    # disagrees with what was built from it.
    note: str | None = None

    def expired(self, on: date) -> bool:
        return self.expires is not None and on > self.expires

    def days_remaining(self, on: date) -> int | None:
        if self.expires is None:
            return None
        return (self.expires - on).days


def _add_months(start: date, months: int) -> date:
    index = start.month - 1 + months
    year = start.year + index // 12
    month = index % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def nasr_effective_date(readme_path: Path | None = None) -> date | None:
    """The cycle date from the first line of the NASR README, or None.
    """
    path = readme_path or NASR_README
    try:
        first_line = path.read_text(errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None

    match = _EFFECTIVE_LINE.search(first_line)
    if not match:
        return None
    month = _MONTHS.get(match.group(1).lower())
    if not month:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def _built_effective_date(db_path: Path | None = None) -> date | None:
    """The cycle the airport database was actually built from."""
    path = db_path or AIRPORT_DB
    if not path.exists():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'effective_date'"
        ).fetchone()
        connection.close()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    try:
        return date.fromisoformat(row[0])
    except ValueError:
        return None


def nasr(
    on: date | None = None,
    *,
    readme_path: Path | None = None,
    db_path: Path | None = None,
) -> Dataset:
    """Currency of the FAA NASR cycle -- one month from the effective date."""
    del on  # kept for symmetry with the callers; expiry is date-independent
    effective = nasr_effective_date(readme_path)
    built = _built_effective_date(db_path)
    note = None

    if effective is None:
        # Fall back to whatever the database was built from, so a missing
        # README degrades to a stale-looking date
        effective = built
        note = (
            "No effective date in data/nasr/README.txt; using the date the "
            "airport database was built from."
            if built
            else "No NASR effective date found."
        )
    elif built and built != effective:
        note = (
            f"Airport database was built from the {built.isoformat()} cycle; "
            f"data/nasr holds {effective.isoformat()}. Re-run "
            f"tools/build_airports.py."
        )

    expires = _add_months(effective, NASR_VALID_MONTHS) if effective else None
    return Dataset("nasr", "FAA NASR", effective, expires, note)


def world_magnetic_model(
    on: date | None = None, *, cof_path: Path | None = None
) -> Dataset:
    """Currency of the WMM coefficients -- five years from the epoch."""
    del on
    try:
        start, end = magnetic.validity_window(cof_path)
    except (OSError, ValueError, IndexError):
        return Dataset(
            "wmm",
            "World Magnetic Model",
            None,
            None,
            "Could not read the WMM coefficient file.",
        )
    return Dataset(
        "wmm",
        "World Magnetic Model",
        _date_for_decimal_year(start),
        _date_for_decimal_year(end),
    )


def _date_for_decimal_year(decimal_year: float) -> date:
    """Inverse of `magnetic.decimal_year_for`, rounded to the nearest day."""
    year = int(decimal_year)
    start = date(year, 1, 1)
    days_in_year = (date(year + 1, 1, 1) - start).days
    return start + timedelta(days=round((decimal_year - year) * days_in_year))


def datasets(on: date | None = None) -> list[Dataset]:
    """Every dated dataset, in the order the UI shows them."""
    return [nasr(on), world_magnetic_model(on)]
