"""Build the local airport database from the FAA NASR subscription.

    make airports                     # finds the newest zip in ~/Downloads
    make airports NASR=/path/to.zip   # or point it at one

Reads the 28-day NASR subscription zip directly -- including the CSV bundle
nested inside it -- so there is nothing to unpack by hand. **No network
access**: unlike the OurAirports build this replaces, the data is a file you
already downloaded, which is the right shape for an offline-first app.

NASR is the authoritative US source and carries what OurAirports could not:
published magnetic variation, traffic pattern altitude, fuel types, tower
type, the sectional each airport appears on, and precise surveyed positions.

Two structural notes about NASR that shape the output:

* **Identifiers are split.** `ARPT_ID` is the FAA identifier (`SQL`) and
  `ICAO_ID` is the ICAO one (`KSQL`), present for only about a fifth of
  airports. The ICAO form is used for display where it exists because that is
  what pilots type, and both are kept searchable.
* **Most airports are private.** Of 12,582 open US airports only 4,730 are
  public use. Both are ingested, flagged by `public_use`. Dropping the private
  ones would discard 7,852 possible forced-landing sites, which is exactly the
  information a planner should keep.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "aero"
DB_PATH = OUT_DIR / "airports.sqlite"
NASR_DIR = ROOT / "data" / "nasr"
# Fallback for a machine that has not unpacked a cycle into data/nasr yet.
DOWNLOADS = Path.home() / "Downloads"

# NASR classification codes we keep. 'A' is a landplane airport; heliports,
# seaplane bases, balloonports, gliderports and ultralight strips are excluded
# because a 172 cannot use them and they would swamp the map.
LANDPLANE = "A"

# NASR keeps every kind of fix in FIX_BASE; this is the code for the published
# VFR waypoints -- the VP-prefixed points printed on sectional charts.
VFR_FIX_USE = "VFR"
OPEN_STATUS = "O"
US = "US"

# Lighting codes that mean the runway is lit at all. NSTD is non-standard
# lighting, which still counts as lit for night planning; the checklist can
# treat it with suspicion, but pretending it is dark would be wrong too.
LIT_CODES = {"HIGH", "MED", "LOW", "FLD", "STRB", "NSTD", "PERI"}


def find_subscription(explicit: str | None) -> Path:
    """Locate the NASR data: an unpacked directory or the subscription zip.

    `data/nasr/` is preferred, since an unpacked cycle checked in beside the
    project is what the app is actually built against. The downloaded zip is
    the fallback for a machine that has not unpacked one.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise SystemExit(f"no NASR subscription at {path}")
        return path
    if (NASR_DIR / "CSV_Data").is_dir():
        return NASR_DIR
    candidates = sorted(
        DOWNLOADS.glob("28DaySubscription_Effective_*.zip"),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"no NASR data found. Unpack a cycle into {NASR_DIR}, or leave a\n"
            f"28DaySubscription_Effective_*.zip in {DOWNLOADS}, or pass --source PATH."
        )
    return candidates[0]


def _csv_bundle(subscription: Path) -> io.BytesIO | Path:
    """The inner `CSV_Data/<date>_CSV.zip`, from a directory or a zip.

    Reading the inner archive from memory avoids unpacking a quarter of a
    gigabyte to disk for the few files actually needed.
    """
    if subscription.is_dir():
        bundles = sorted((subscription / "CSV_Data").glob("*_CSV.zip"))
        if not bundles:
            raise SystemExit(
                f"{subscription} has no CSV_Data/*_CSV.zip; this build needs the "
                f"CSV flavour of NASR, not the fixed-width .txt files"
            )
        return bundles[-1]

    with zipfile.ZipFile(subscription) as outer:
        inner_names = [
            n for n in outer.namelist() if n.startswith("CSV_Data/") and n.endswith(".zip")
        ]
        if not inner_names:
            raise SystemExit(
                f"{subscription.name} has no CSV_Data bundle; this build needs the "
                f"CSV flavour of NASR, not the fixed-width .txt files"
            )
        with outer.open(inner_names[0]) as handle:
            return io.BytesIO(handle.read())


def _csv_bundle_name(subscription: Path) -> str:
    """A name that identifies the cycle, for the recorded provenance."""
    if subscription.is_dir():
        bundles = sorted((subscription / "CSV_Data").glob("*_CSV.zip"))
        if bundles:
            return bundles[-1].name
    return subscription.name


def read_nasr_csvs(subscription: Path, wanted: set[str]) -> dict[str, list[dict]]:
    """Pull named CSVs out of the CSV bundle nested inside the subscription."""
    out: dict[str, list[dict]] = {}
    with zipfile.ZipFile(_csv_bundle(subscription)) as inner:
        available = set(inner.namelist())
        for name in wanted:
            if name not in available:
                raise SystemExit(f"{name} missing from the NASR CSV bundle")
            with inner.open(name) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8-sig", errors="replace")
                out[name] = list(csv.DictReader(text))
    return out


_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}


def effective_date(subscription: Path) -> str | None:
    """The cycle date, as ISO, so an expired cycle is visible at a glance.

    The FAA names a cycle two ways depending on which artefact you have:
    `28DaySubscription_Effective_2026-06-11.zip` for the subscription, and
    `11_Jun_2026_CSV.zip` for the bundle inside it. Both are read, since
    either may be what is on disk.
    """
    names = [subscription.name]
    if subscription.is_dir():
        names += [p.name for p in sorted((subscription / "CSV_Data").glob("*_CSV.zip"))]

    for name in names:
        iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
        if iso:
            return iso.group(0)
        worded = re.search(r"(\d{1,2})_([A-Za-z]{3})[a-z]*_(\d{4})", name)
        if worded:
            day, month, year = worded.groups()
            number = _MONTHS.get(month.lower())
            if number:
                return f"{year}-{number:02d}-{int(day):02d}"
    return None


def to_float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def clean(value: str | None) -> str | None:
    """Empty strings become NULL, so 'no data' is distinguishable from ''."""
    text = (value or "").strip()
    return text or None


def signed_variation(row: dict) -> float | None:
    """Published magnetic variation as a signed number, east positive.

    NASR stores the magnitude and hemisphere separately. Aviation convention
    and `engine.magnetic` both treat east as positive, so west becomes
    negative here rather than at every call site.
    """
    magnitude = to_float(row.get("MAG_VARN"))
    if magnitude is None:
        return None
    return -magnitude if (row.get("MAG_HEMIS") or "").strip().upper() == "W" else magnitude


def classify(towered: bool, longest_ft: float | None) -> str:
    """A size class, since NASR does not publish one.

    Only used for display ordering -- showing the significant airports first
    when a query is truncated. The underlying facts are stored separately, so
    nothing decides anything real from this.
    """
    length = longest_ft or 0.0
    if towered and length >= 8000:
        return "large_airport"
    if towered or length >= 5000:
        return "medium_airport"
    return "small_airport"


def build(subscription: Path) -> int:
    print(f"Reading {subscription.name}")
    tables = read_nasr_csvs(subscription, {"APT_BASE.csv", "APT_RWY.csv", "FIX_BASE.csv"})
    base, runways = tables["APT_BASE.csv"], tables["APT_RWY.csv"]
    fixes = tables.get("FIX_BASE.csv", [])
    print(f"  APT_BASE {len(base)} rows, APT_RWY {len(runways)} rows, FIX_BASE {len(fixes)} rows")

    print("\nFiltering:")
    airports = [
        row
        for row in base
        if row["SITE_TYPE_CODE"] == LANDPLANE
        and row["ARPT_STATUS"] == OPEN_STATUS
        and row["COUNTRY_CODE"] == US
    ]
    print(f"  {len(airports)} open US landplane airports")
    public = sum(1 for r in airports if r["FACILITY_USE_CODE"] == "PU")
    print(f"    {public} public use, {len(airports) - public} private")

    # A departure or destination needs an elevation for climb and descent.
    # NASR publishes one for every airport, but check rather than assume.
    missing = [r for r in airports if to_float(r["ELEV"]) is None]
    if missing:
        print(f"  dropping {len(missing)} with no published elevation")
        airports = [r for r in airports if to_float(r["ELEV"]) is not None]

    # Runways join on SITE_NO, not the identifier: identifiers are reassigned
    # over time and are not a stable key, whereas the site number is.
    keep_sites = {r["SITE_NO"] for r in airports}
    runways = [r for r in runways if r["SITE_NO"] in keep_sites]
    print(f"  {len(runways)} runways on those airports")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    connection = sqlite3.connect(DB_PATH)
    connection.executescript(SCHEMA)

    def ident_of(row: dict) -> str:
        return clean(row["ICAO_ID"]) or row["ARPT_ID"].strip()

    connection.executemany(
        """
        INSERT OR IGNORE INTO airports (
            ident, icao, iata, faa_ident, site_no, name, kind, lat, lon,
            elevation_ft, municipality, region, public_use, towered,
            pattern_altitude_ft, fuel_types, magnetic_variation_deg,
            variation_year, sectional, artcc, flight_service
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                ident_of(row),
                clean(row["ICAO_ID"]),
                None,  # NASR carries no IATA code
                row["ARPT_ID"].strip(),
                row["SITE_NO"],
                (row["ARPT_NAME"] or "").strip().title(),
                "small_airport",  # replaced below, once runways are known
                to_float(row["LAT_DECIMAL"]),
                to_float(row["LONG_DECIMAL"]),
                to_float(row["ELEV"]),
                (row["CITY"] or "").strip().title() or None,
                f"US-{row['STATE_CODE']}" if clean(row["STATE_CODE"]) else None,
                1 if row["FACILITY_USE_CODE"] == "PU" else 0,
                1 if (row["TWR_TYPE_CODE"] or "").startswith("ATCT") else 0,
                to_float(row["TPA"]),
                clean(row["FUEL_TYPES"]),
                signed_variation(row),
                to_float(row["MAG_VARN_YEAR"]),
                clean(row["CHART_NAME"]),
                clean(row["RESP_ARTCC_ID"]),
                clean(row["FSS_ID"]),
            )
            for row in airports
        ],
    )

    site_to_ident = {row["SITE_NO"]: ident_of(row) for row in airports}
    connection.executemany(
        """
        INSERT INTO runways
            (airport_ident, designation, length_ft, width_ft, surface, lighted)
        VALUES (?,?,?,?,?,?)
        """,
        [
            (
                site_to_ident[row["SITE_NO"]],
                (row["RWY_ID"] or "").strip(),
                to_float(row["RWY_LEN"]),
                to_float(row["RWY_WIDTH"]),
                clean(row["SURFACE_TYPE_CODE"]),
                1 if (row["RWY_LGT_CODE"] or "").strip().upper() in LIT_CODES else 0,
            )
            for row in runways
        ],
    )

    # Denormalise the longest runway onto the airport. The go/no-go check asks
    # "can I land here" far more often than it asks about one runway.
    connection.execute(
        """
        UPDATE airports SET longest_runway_ft = (
            SELECT MAX(length_ft) FROM runways
             WHERE runways.airport_ident = airports.ident
        )
        """
    )
    connection.execute(
        """
        UPDATE airports SET kind = CASE
            WHEN towered = 1 AND COALESCE(longest_runway_ft, 0) >= 8000 THEN 'large_airport'
            WHEN towered = 1 OR COALESCE(longest_runway_ft, 0) >= 5000 THEN 'medium_airport'
            ELSE 'small_airport' END
        """
    )
    # VFR waypoints. NASR carries every kind of fix in one file; the VFR ones
    # are the VP-prefixed points printed on sectionals, and the only kind a
    # VFR planner should offer.
    vfr = [
        row
        for row in fixes
        if row.get("FIX_USE_CODE", "").strip() == VFR_FIX_USE
        and row.get("COUNTRY_CODE", "").strip() in ("", US)
        and to_float(row.get("LAT_DECIMAL")) is not None
        and to_float(row.get("LONG_DECIMAL")) is not None
    ]
    print(f"  {len(vfr)} published VFR waypoints")
    connection.executemany(
        "INSERT OR IGNORE INTO vfr_waypoints (ident, lat, lon, state, artcc) VALUES (?,?,?,?,?)",
        [
            (
                row["FIX_ID"].strip(),
                to_float(row["LAT_DECIMAL"]),
                to_float(row["LONG_DECIMAL"]),
                clean(row.get("STATE_CODE")),
                clean(row.get("ARTCC_ID_LOW")),
            )
            for row in vfr
        ],
    )

    connection.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        [
            # Name the cycle bundle, not the directory it was unpacked into:
            # "nasr" says nothing about which 28 days this is.
            ("source", _csv_bundle_name(subscription)),
            # The 28-day cycle this data is valid for. Carried through to the
            # UI and the printed navlog: a planner working from an expired
            # cycle should be able to see that at a glance.
            ("effective_date", effective_date(subscription) or ""),
        ],
    )
    connection.commit()

    report(connection)
    connection.close()
    print(f"\nWrote {DB_PATH.relative_to(ROOT)} ({DB_PATH.stat().st_size // 1024} KB)")
    print(f"Source: FAA NASR, {subscription.name}. Public domain.")
    return 0


def report(connection: sqlite3.Connection) -> None:
    def scalar(sql: str) -> int:
        return connection.execute(sql).fetchone()[0]

    total = scalar("SELECT COUNT(*) FROM airports")
    print(f"\n{total} airports:")
    for kind, count in connection.execute(
        "SELECT kind, COUNT(*) FROM airports GROUP BY kind ORDER BY 2 DESC"
    ):
        print(f"  {kind:<16} {count}")
    print(f"  {scalar('SELECT COUNT(*) FROM airports WHERE public_use = 1')} public use")
    print(f"  {scalar('SELECT COUNT(*) FROM airports WHERE towered = 1')} towered")
    print(
        f"  {scalar('SELECT COUNT(*) FROM airports WHERE longest_runway_ft IS NOT NULL')}"
        f" with runway length"
    )
    print(f"  {scalar('SELECT COUNT(*) FROM airports WHERE fuel_types IS NOT NULL')} with fuel")
    print(
        f"  {scalar('SELECT COUNT(*) FROM airports WHERE pattern_altitude_ft IS NOT NULL')}"
        f" with a published pattern altitude"
    )
    print(f"  {scalar('SELECT COUNT(*) FROM runways')} runways")


SCHEMA = """
CREATE TABLE airports (
    ident                  TEXT PRIMARY KEY,   -- ICAO where published, else FAA
    icao                   TEXT,
    iata                   TEXT,               -- always NULL; NASR has none
    faa_ident              TEXT NOT NULL,      -- 'SQL' where ident is 'KSQL'
    site_no                TEXT NOT NULL,      -- NASR's stable key
    name                   TEXT NOT NULL,
    kind                   TEXT NOT NULL,      -- derived, for display order only
    lat                    REAL NOT NULL,
    lon                    REAL NOT NULL,
    elevation_ft           REAL NOT NULL,
    municipality           TEXT,
    region                 TEXT,
    public_use             INTEGER NOT NULL,
    towered                INTEGER NOT NULL,
    pattern_altitude_ft    REAL,
    fuel_types             TEXT,
    magnetic_variation_deg REAL,               -- east positive
    variation_year         REAL,
    sectional              TEXT,
    artcc                  TEXT,
    flight_service         TEXT,
    longest_runway_ft      REAL
);
CREATE INDEX airports_position ON airports (lat, lon);
CREATE INDEX airports_name ON airports (name);
CREATE INDEX airports_faa_ident ON airports (faa_ident);

CREATE TABLE runways (
    airport_ident TEXT NOT NULL REFERENCES airports (ident),
    designation   TEXT,
    length_ft     REAL,
    width_ft      REAL,
    surface       TEXT,
    lighted       INTEGER
);
CREATE INDEX runways_airport ON runways (airport_ident);

-- Published VFR waypoints, from NASR's fix file. These are the five-letter
-- VP-prefixed points printed on sectionals -- bridges, dams, reporting points
-- -- and they are exactly what a VFR route should be built from, because they
-- are far easier to identify from the air than an arbitrary lat/lon.
CREATE TABLE vfr_waypoints (
    ident  TEXT PRIMARY KEY,
    lat    REAL NOT NULL,
    lon    REAL NOT NULL,
    state  TEXT,
    artcc  TEXT
);
CREATE INDEX vfr_waypoints_bbox ON vfr_waypoints (lat, lon);

CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", help="path to a 28DaySubscription_Effective_*.zip", default=None
    )
    args = parser.parse_args()
    return build(find_subscription(args.source))


if __name__ == "__main__":
    sys.exit(main())
