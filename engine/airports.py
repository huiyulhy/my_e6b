"""Airport lookup against the local database - this needs to be built 
by running `tools/build_airports.py`.

Reads `data/aero/airports.sqlite`
SQLite is used rather than an in-memory structure because it is what the app
ships with: one file, queryable, and readable unchanged by a later C++ port.
The connection is opened read-only and cached, so repeated lookups during a
route search do not reopen it.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from engine import preflight
from engine.geo import LatLon, inverse

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "aero" / "airports.sqlite"

# Degrees of latitude per nautical mile, for bounding a search before the
# exact geodesic distance is computed. Longitude is corrected by latitude.
_DEG_PER_NM = 1.0 / 60.0


class AirportDatabaseMissing(FileNotFoundError):
    """The airport database has not been built yet."""


@dataclass(frozen=True)
class Airport:
    ident: str
    name: str
    kind: str
    position: LatLon
    elevation_ft: float
    icao: str | None = None
    iata: str | None = None
    municipality: str | None = None
    region: str | None = None
    longest_runway_ft: float | None = None
    # The published traffic pattern altitude, in feet AGL -- NASR gives a
    # height above the field, not an MSL altitude. `None` for most fields:
    # NASR only carries one where it is non-standard or somebody filed it, so
    # the go/no-go falls back to the standard 1,000 ft rather than assuming
    # the absence means anything.
    pattern_altitude_agl_ft: float | None = None

    @property
    def label(self) -> str:
        where = f", {self.municipality}" if self.municipality else ""
        return f"{self.ident} - {self.name}{where}"


# One connection per thread, not one shared between them.
#
# A `sqlite3.Connection` is not safe for simultaneous use, and
# `check_same_thread=False` only removes Python's guard against sharing -- it
# does not make sharing correct. A single cached connection used from several
# threads at once interleaves results on its cursor, which shows up as
# `InterfaceError: bad parameter or other API misuse` if you are lucky and as a
# row with somebody else's columns in it if you are not. The second is the real
# hazard: a lookup that quietly returns the wrong airport's position.
#
# It went unnoticed while every request was serial. The weather panel fetches
# every field on the route at once, each worker resolving an identifier, so the
# lookups genuinely overlap now.
#
# Thread-local rather than a lock, because the database is opened read-only and
# SQLite handles concurrent readers on separate connections perfectly well;
# serialising them would give back the parallelism the fetch exists for.
# Connections are closed by the interpreter at exit, and threads here come from
# a bounded pool, so the map cannot grow without bound.
_local = threading.local()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    db = path or DB_PATH
    connections = getattr(_local, "connections", None)
    if connections is None:
        connections = _local.connections = {}

    key = str(db)
    connection = connections.get(key)
    if connection is None:
        if not db.exists():
            raise AirportDatabaseMissing(
                f"{db} not found. Build it with `make airports`."
            )
        # `check_same_thread` left at its default: each thread owns its own
        # connection now, so the guard is a safety net rather than an obstacle.
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connections[key] = connection
    return connection


def _to_airport(row: sqlite3.Row) -> Airport:
    return Airport(
        ident=row["ident"],
        name=row["name"],
        kind=row["kind"],
        position=LatLon(row["lat"], row["lon"]),
        elevation_ft=row["elevation_ft"],
        icao=row["icao"],
        iata=row["iata"],
        municipality=row["municipality"],
        region=row["region"],
        longest_runway_ft=row["longest_runway_ft"],
        pattern_altitude_agl_ft=row["pattern_altitude_ft"],
    )


def runways(ident: str, *, path: Path | None = None) -> list[preflight.Runway]:
    """Every runway on a field, longest first.

    Returned as `preflight.Runway` rather than a type of our own, because the
    only consumer is the go/no-go check and a second near-identical record
    would just need converting at every call site. Rows with no published
    length still come back -- the checklist reports an unknown runway as
    unknown, which is not the same as it not existing.
    """
    rows = _connect(path).execute(
        """
        SELECT designation, length_ft, surface, lighted
          FROM runways
         WHERE airport_ident = ?
         ORDER BY COALESCE(length_ft, 0) DESC
        """,
        (ident,),
    ).fetchall()
    return [
        preflight.Runway(
            designation=row["designation"] or "",
            length_ft=row["length_ft"],
            surface=row["surface"] or "",
            lighted=bool(row["lighted"]),
        )
        for row in rows
    ]


def find(ident: str, *, path: Path | None = None) -> Airport | None:
    """Look an airport up by identifier.

    Accepts the local identifier, the ICAO code, or the IATA code
    """
    key = ident.strip().upper()
    row = _connect(path).execute(
        "SELECT * FROM airports WHERE ident = ? OR icao = ? OR iata = ? LIMIT 1",
        (key, key, key),
    ).fetchone()
    return _to_airport(row) if row else None


def search(query: str, limit: int = 20, *, path: Path | None = None) -> list[Airport]:
    """Free-text search over identifier, name and city.

    Exact identifier matches are ranked first, then prefix matches on the
    identifier, then everything else -- so typing "SQL" finds San Carlos
    before it finds airports with "sql" buried in a name.
    """
    text = query.strip().upper()
    if not text:
        return []
    like = f"%{text}%"
    rows = _connect(path).execute(
        """
        SELECT *,
            CASE
                WHEN ident = ?1 OR icao = ?1 THEN 0
                WHEN ident LIKE ?1 || '%' OR icao LIKE ?1 || '%' THEN 1
                WHEN UPPER(name) LIKE ?2 THEN 2
                ELSE 3
            END AS rank
        FROM airports
        WHERE ident LIKE ?2 OR icao LIKE ?2 OR iata = ?1
           OR UPPER(name) LIKE ?2 OR UPPER(municipality) LIKE ?2
        ORDER BY rank,
                 CASE kind WHEN 'large_airport' THEN 0
                           WHEN 'medium_airport' THEN 1 ELSE 2 END,
                 ident
        LIMIT ?3
        """,
        (text, like, limit),
    ).fetchall()
    return [_to_airport(row) for row in rows]


def in_bounding_box(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    limit: int = 2000,
    min_runway_ft: float | None = None,
    path: Path | None = None,
) -> list[Airport]:
    """Airports within a lat/lon box, for drawing the visible map area.

    Larger airports are returned first so that a truncated result still shows
    the ones worth seeing at low zoom.
    """
    clause = "AND longest_runway_ft >= ?" if min_runway_ft is not None else ""
    params: list[float] = [south, north, west, east]
    if min_runway_ft is not None:
        params.append(min_runway_ft)
    params.append(limit)
    rows = _connect(path).execute(
        f"""
        SELECT * FROM airports
        WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? {clause}
        ORDER BY CASE kind WHEN 'large_airport' THEN 0
                           WHEN 'medium_airport' THEN 1 ELSE 2 END,
                 COALESCE(longest_runway_ft, 0) DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_to_airport(row) for row in rows]


def near(
    position: LatLon,
    radius_nm: float,
    *,
    limit: int = 50,
    min_runway_ft: float | None = None,
    path: Path | None = None,
) -> list[Airport]:
    """Airports within a radius, nearest first.

    A bounding box narrows the candidates in SQL, then the exact geodesic
    distance is computed in Python. Doing it the other way round would mean a
    great-circle solve for every airport in the country.
    """
    lat_margin = radius_nm * _DEG_PER_NM
    # Guard the cosine near the poles, where longitude degrees collapse.
    from math import cos, radians

    scale = max(cos(radians(position.lat)), 0.01)
    lon_margin = lat_margin / scale

    candidates = in_bounding_box(
        position.lat - lat_margin,
        position.lon - lon_margin,
        position.lat + lat_margin,
        position.lon + lon_margin,
        limit=5000,
        min_runway_ft=min_runway_ft,
        path=path,
    )
    within = [
        (inverse(position, airport.position).distance_nm, airport)
        for airport in candidates
    ]
    within = [pair for pair in within if pair[0] <= radius_nm]
    within.sort(key=lambda pair: pair[0])
    return [airport for _, airport in within[:limit]]


def count(*, path: Path | None = None) -> int:
    return _connect(path).execute("SELECT COUNT(*) FROM airports").fetchone()[0]


# --- published VFR waypoints ---------------------------------------------
#
# The VP-prefixed points printed on sectionals: a reservoir, a bridge, a
# racetrack. They matter more than their small number suggests, because a
# route built from them is one a pilot can actually fly by looking out of the
# window. "Track to VPLEX over Lexington Reservoir" is an instruction;
# "track to 37.19 N 121.99 W" is not.


@dataclass(frozen=True)
class VfrWaypoint:
    ident: str
    position: LatLon
    state: str | None = None
    artcc: str | None = None

    @property
    def label(self) -> str:
        where = f" ({self.state})" if self.state else ""
        return f"{self.ident}{where}"


def _to_vfr_waypoint(row: sqlite3.Row) -> VfrWaypoint:
    return VfrWaypoint(
        ident=row["ident"],
        position=LatLon(row["lat"], row["lon"]),
        state=row["state"],
        artcc=row["artcc"],
    )


def find_vfr_waypoint(ident: str, *, path: Path | None = None) -> VfrWaypoint | None:
    """Look a VFR waypoint up by identifier.

    The `VP` prefix is optional, so `LEX` and `VPLEX` both resolve -- pilots
    read them off a chart either way.
    """
    key = ident.strip().upper()
    row = _connect(path).execute(
        "SELECT * FROM vfr_waypoints WHERE ident = ? OR ident = 'VP' || ? LIMIT 1",
        (key, key),
    ).fetchone()
    return _to_vfr_waypoint(row) if row else None


def search_vfr_waypoints(
    query: str, limit: int = 10, *, path: Path | None = None
) -> list[VfrWaypoint]:
    """Identifier search. There is nothing else to match on.

    Unlike airports these carry no name or city in NASR -- only the code, its
    position and the controlling centre. What the point actually depicts is
    printed on the sectional, not published in the data.
    """
    text = query.strip().upper()
    if not text:
        return []
    rows = _connect(path).execute(
        """
        SELECT * FROM vfr_waypoints
         WHERE ident LIKE ?1 || '%' OR ident LIKE '%' || ?1 || '%'
         ORDER BY CASE WHEN ident = ?1 OR ident = 'VP' || ?1 THEN 0
                       WHEN ident LIKE ?1 || '%' THEN 1 ELSE 2 END,
                  ident
         LIMIT ?2
        """,
        (text, limit),
    ).fetchall()
    return [_to_vfr_waypoint(row) for row in rows]


def vfr_waypoints_in_bounding_box(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    limit: int = 500,
    path: Path | None = None,
) -> list[VfrWaypoint]:
    """VFR waypoints in the visible map area."""
    rows = _connect(path).execute(
        """
        SELECT * FROM vfr_waypoints
         WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
         ORDER BY ident LIMIT ?
        """,
        (south, north, west, east, limit),
    ).fetchall()
    return [_to_vfr_waypoint(row) for row in rows]


def vfr_waypoints_near(
    position: LatLon,
    radius_nm: float,
    *,
    limit: int = 50,
    path: Path | None = None,
) -> list[VfrWaypoint]:
    """VFR waypoints within a radius, nearest first.
    """
    from math import cos, radians

    lat_margin = radius_nm * _DEG_PER_NM
    scale = max(cos(radians(position.lat)), 0.01)
    lon_margin = lat_margin / scale

    candidates = vfr_waypoints_in_bounding_box(
        position.lat - lat_margin,
        position.lon - lon_margin,
        position.lat + lat_margin,
        position.lon + lon_margin,
        limit=2000,
        path=path,
    )
    within = [
        (inverse(position, waypoint.position).distance_nm, waypoint)
        for waypoint in candidates
    ]
    within = [pair for pair in within if pair[0] <= radius_nm]
    within.sort(key=lambda pair: pair[0])
    return [waypoint for _, waypoint in within[:limit]]


def count_vfr_waypoints(*, path: Path | None = None) -> int:
    return _connect(path).execute("SELECT COUNT(*) FROM vfr_waypoints").fetchone()[0]
