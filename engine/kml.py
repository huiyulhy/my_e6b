"""The route as a KML file, and a KML file back into a route.

Stores the mission as a KML format to be reloaded
constraints:
- LLA (BOC or BOD or TOC or TOD)
- What the planner needs to rebuild: weather per segment
"""

from __future__ import annotations

import io
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from typing import Any

from engine import mission as ms
from engine.atmosphere import FT_PER_M, M_PER_FT
from engine.geo import MAX_LATITUDE_DEG
from engine.navlog import Navlog, Waypoint

__all__ = [
    "ImportedWaypoint",
    "KmlError",
    "KmlFile",
    "kml_filename",
    "mission_kml",
    "parse_kml",
    "read_kml",
    "route_kml",
]

KML_NS = "http://www.opengis.net/kml/2.2"

# Our `ExtendedData` names, prefixed so they cannot collide with a field some
# other program writes under the same plain name.
DATA_PREFIX = "e6b:"

# A route is a few dozen points. A file larger than this is not one, and an
# XML parser handed a large enough document is a denial of service on its own.
MAX_BYTES = 5 * 1024 * 1024

AIRPORT_KINDS = ("airport", "large_airport", "medium_airport", "small_airport")


class KmlError(ValueError):
    """The file is not a route this program can read."""


@dataclass(frozen=True)
class ImportedWaypoint:
    """One point read out of a file, before the server or UI has decided
    anything about it.

    Not an `engine.profile.Waypoint`: that wants a position object, runways
    and the rest, and this is only what the file said. `altitude_ft` is the
    crossing constraint when the file was one of ours and the coordinate's own
    altitude otherwise; `from_extended_data` says which.
    """

    name: str
    lat: float
    lon: float
    altitude_ft: float | None = None
    kind: str = "waypoint"
    segment_type: str = "automatic"
    is_landing: bool = False
    elevation_ft: float | None = None
    from_extended_data: bool = False


@dataclass(frozen=True)
class KmlFile:
    """Everything read out of a file: the route, and our mission if it has one.

    `mission` is the plan request the file was exported from, and `snapshot`
    the solve it was exported with. A mission this version cannot read is not
    fatal -- the placemarks are still a route -- so it comes back as
    `mission_error` beside the waypoints rather than as a refusal.
    """

    waypoints: list[ImportedWaypoint]
    mission: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


# --- export ----------------------------------------------------------------


def route_kml(
    navlog: Navlog, *, mission: str | None = None, snapshot: str | None = None
) -> str:
    """The resolved route as a KML document: a placemark per waypoint at its
    planned altitude, and the route as a line through them.

    `mission` and `snapshot` are `engine.mission` JSON, carried in the
    document's own `ExtendedData` when given."""
    points = list(navlog.resolved_waypoints)
    if not points:
        raise KmlError("the plan has no waypoints to export")
    return _document(points, _planned_altitudes(navlog), mission=mission, snapshot=snapshot)


def mission_kml(waypoints: list[Waypoint], *, mission: str) -> str:
    """The pilot's own points and the mission, for a plan that does not solve.

    A mission that cannot be planned today -- a leg too short for its climb,
    a wind too strong -- is still worth saving: tomorrow's weather may fly it.
    With no profile the points sit at their crossing altitude where they have
    one and on the ground where they do not.
    """
    points = [w for w in waypoints if not w.generated]
    if not points:
        raise KmlError("the mission has no waypoints to export")
    altitudes = [
        w.altitude_ft if w.altitude_ft is not None else (w.elevation_ft or 0.0) for w in points
    ]
    return _document(points, altitudes, mission=mission, snapshot=None)


def _document(
    points: list[Waypoint],
    altitudes: list[float],
    *,
    mission: str | None,
    snapshot: str | None,
) -> str:
    ET.register_namespace("", KML_NS)
    root = ET.Element(_tag("kml"))
    document = _child(root, "Document")
    _child(document, "name", f"{points[0].name} to {points[-1].name}")
    if mission is not None or snapshot is not None:
        data = _child(document, "ExtendedData")
        for key, value in (("mission", mission), ("snapshot", snapshot)):
            if value is not None:
                _child(_child(data, "Data", name=DATA_PREFIX + key), "value", value)
    for style, colour in (("airport", "ff2ad4ff"), ("waypoint", "ffffffff"), ("phase", "ff9a9a9a")):
        node = _child(document, "Style", id=style)
        icon = _child(node, "IconStyle")
        _child(icon, "color", colour)  # KML colours are aabbggrr
        _child(icon, "scale", "0.9")

    coordinates = []
    for waypoint, altitude_ft in zip(points, altitudes, strict=True):
        coordinate = f"{waypoint.position.lon:.6f},{waypoint.position.lat:.6f},{altitude_ft * M_PER_FT:.1f}"
        coordinates.append(coordinate)
        placemark = _child(document, "Placemark")
        _child(placemark, "name", waypoint.name)
        _child(placemark, "description", _description(waypoint, altitude_ft))
        _child(placemark, "styleUrl", f"#{_style_for(waypoint)}")
        data = _child(placemark, "ExtendedData")
        for key, value in (
            ("kind", waypoint.kind),
            ("segment_type", waypoint.segment_type),
            ("altitude_ft", _number(waypoint.altitude_ft)),
            ("elevation_ft", _number(waypoint.elevation_ft)),
            ("is_landing", "1" if waypoint.is_landing else "0"),
            ("generated", "1" if waypoint.generated else "0"),
        ):
            _child(_child(data, "Data", name=DATA_PREFIX + key), "value", value)
        point = _child(placemark, "Point")
        _child(point, "altitudeMode", "absolute")
        _child(point, "coordinates", coordinate)

    path = _child(document, "Placemark")
    _child(path, "name", "Route")
    line = _child(path, "LineString")
    _child(line, "tessellate", "1")
    _child(line, "altitudeMode", "absolute")
    _child(line, "coordinates", " ".join(coordinates))

    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def kml_filename(navlog: Navlog) -> str:
    """`route-KSQL-KSBP.kml`, or `route.kml` if the route has no names."""
    names = [w.name for w in navlog.resolved_waypoints]
    parts = [_safe(name) for name in (names[0], names[-1])] if names else []
    return "-".join(["route", *(p for p in parts if p)]) + ".kml"


def _planned_altitudes(navlog: Navlog) -> list[float]:
    """The altitude the plan has the aeroplane at over each resolved waypoint.

    Read off the legs rather than the waypoints: a waypoint carries only its
    constraint, if any, and it is the leg leaving it that knows what altitude
    the profile actually passes it at. The legs and the waypoints are walked
    together, so a route that visits the same field twice keeps each visit's
    own altitude.
    """
    points = navlog.resolved_waypoints
    altitudes: list[float | None] = [None] * len(points)
    at = 0
    for leg in navlog.legs:
        if not leg.covers_ground:
            continue
        start = next((i for i in range(at, len(points)) if points[i].name == leg.from_name), None)
        if start is None:
            continue
        at = start
        if altitudes[at] is None and leg.entry_altitude_ft is not None:
            altitudes[at] = leg.entry_altitude_ft
        if at + 1 < len(points) and points[at + 1].name == leg.to_name:
            if leg.exit_altitude_ft is not None:
                altitudes[at + 1] = leg.exit_altitude_ft
            at += 1
    # A point no leg described -- a single-point route, or a row that never
    # left the ground -- is placed at its constraint, failing that its ground.
    return [
        alt if alt is not None else (w.altitude_ft if w.altitude_ft is not None else (w.elevation_ft or 0.0))
        for alt, w in zip(altitudes, points, strict=True)
    ]


def _description(waypoint: Waypoint, altitude_ft: float) -> str:
    lines = [f"Planned altitude {altitude_ft:.0f} ft MSL"]
    if waypoint.elevation_ft is not None:
        lines.append(f"Elevation {waypoint.elevation_ft:.0f} ft")
    if waypoint.altitude_ft is not None:
        lines.append(f"Cross at {waypoint.altitude_ft:.0f} ft")
    if waypoint.generated:
        lines.append("Inserted by the planner")
    return "\n".join(lines)


def _style_for(waypoint: Waypoint) -> str:
    if waypoint.generated:
        return "phase"
    return "airport" if waypoint.kind in AIRPORT_KINDS else "waypoint"


def _tag(name: str) -> str:
    return f"{{{KML_NS}}}{name}"


def _child(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, _tag(tag), attrs)
    if text is not None:
        node.text = text
    return node


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _safe(name: str) -> str:
    """A waypoint name reduced to something a filesystem will take."""
    return re.sub(r"[^A-Za-z0-9]+", "", name)[:12]


# --- import ----------------------------------------------------------------


def parse_kml(data: bytes) -> list[ImportedWaypoint]:
    """The waypoints in a `.kml` or `.kmz` file, in document order.

    Placemarks with a point are the waypoints. A file with none but with a
    line is a waypoint per vertex of the first line, named `WP1`, `WP2`...
    Points the planner itself inserted are dropped: it will insert them again.
    """
    return read_kml(data).waypoints


def read_kml(data: bytes) -> KmlFile:
    """`parse_kml`, plus the mission and snapshot if the file carries them."""
    root = _parse(data)
    waypoints = [w for w in _placemark_points(root) if w is not None]
    if not waypoints:
        waypoints = _line_points(root)
    if not waypoints:
        raise KmlError("no placemarks or paths found in the file")
    for waypoint in waypoints:
        _check(waypoint)

    stored = _document_data(root)
    mission = snapshot = None
    warnings: list[str] = []
    if "mission" in stored:
        try:
            mission = ms.parse_mission(stored["mission"])
        except ms.MissionError as exc:
            warnings.append(f"{exc}; only the route was imported.")
    if mission is not None and "snapshot" in stored:
        try:
            snapshot = ms.parse_snapshot(stored["snapshot"])
        except ms.MissionError:
            # Only ever used to say what changed; without it nothing is lost.
            snapshot = None
    return KmlFile(waypoints=waypoints, mission=mission, snapshot=snapshot, warnings=warnings)


def _parse(data: bytes) -> ET.Element:
    text = _unwrap(data)
    if len(text) > MAX_BYTES:
        raise KmlError(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
    # Refused rather than parsed with entities disabled: expat's entity
    # handling cannot be turned off from ElementTree, and nothing that writes
    # KML writes a DOCTYPE.
    if re.search(rb"<!\s*DOCTYPE", text, re.IGNORECASE):
        raise KmlError("file declares a document type, which KML never does")
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise KmlError(f"not well-formed XML: {exc}") from None


def _document_data(root: ET.Element) -> dict[str, str]:
    """Our fields on the `Document` itself, as opposed to on a placemark."""
    document = next((n for n in root.iter() if _local(n) == "Document"), None)
    return {} if document is None else _extended_data(document)


def _unwrap(data: bytes) -> bytes:
    """The KML text inside a KMZ, or the bytes as given."""
    if not data.startswith(b"PK"):
        return data
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith(".kml")]
            if not members:
                raise KmlError("the KMZ archive holds no .kml file")
            # The declared size is checked before anything is inflated: a
            # small archive can declare a very large file.
            if members[0].file_size > MAX_BYTES:
                raise KmlError(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
            return archive.read(members[0])
    except zipfile.BadZipFile:
        raise KmlError("not a KMZ archive") from None


def _local(node: ET.Element) -> str:
    """The tag without its namespace, so 2.1, 2.2 and gx files read alike."""
    return node.tag.rsplit("}", 1)[-1]


def _find(node: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in node if _local(c) == name), None)


def _text(node: ET.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _placemark_points(root: ET.Element) -> list[ImportedWaypoint | None]:
    found: list[ImportedWaypoint | None] = []
    unnamed = 0
    for placemark in (n for n in root.iter() if _local(n) == "Placemark"):
        point = next((n for n in placemark.iter() if _local(n) == "Point"), None)
        if point is None:
            continue
        coordinates = _coordinates(_text(_find(point, "coordinates")))
        if not coordinates:
            continue
        lon, lat, alt_m = coordinates[0]
        name = _text(_find(placemark, "name"))
        if not name:
            unnamed += 1
            name = f"WP{unnamed}"
        extended = _extended_data(placemark)
        if extended:
            if extended.get("generated") == "1":
                found.append(None)
                continue
            found.append(
                ImportedWaypoint(
                    name=name,
                    lat=lat,
                    lon=lon,
                    altitude_ft=_float(extended.get("altitude_ft")),
                    kind=extended.get("kind") or "waypoint",
                    segment_type=extended.get("segment_type") or "automatic",
                    is_landing=extended.get("is_landing") == "1",
                    elevation_ft=_float(extended.get("elevation_ft")),
                    from_extended_data=True,
                )
            )
        else:
            found.append(ImportedWaypoint(name=name, lat=lat, lon=lon, altitude_ft=_altitude(alt_m)))
    return found


def _line_points(root: ET.Element) -> list[ImportedWaypoint]:
    for line in (n for n in root.iter() if _local(n) in ("LineString", "LinearRing")):
        coordinates = _coordinates(_text(_find(line, "coordinates")))
        if coordinates:
            return [
                ImportedWaypoint(name=f"WP{i}", lat=lat, lon=lon, altitude_ft=_altitude(alt_m))
                for i, (lon, lat, alt_m) in enumerate(coordinates, start=1)
            ]
    # A Google Earth recorded track: one <gx:coord> per fix, space separated.
    for track in (n for n in root.iter() if _local(n) == "Track"):
        fixes = [c for c in track if _local(c) == "coord"]
        points = [_triple(_text(c).replace(" ", ",")) for c in fixes]
        if points:
            return [
                ImportedWaypoint(name=f"WP{i}", lat=lat, lon=lon, altitude_ft=_altitude(alt_m))
                for i, (lon, lat, alt_m) in enumerate(points, start=1)
            ]
    return []


def _extended_data(placemark: ET.Element) -> dict[str, str]:
    """Our own fields, if the placemark carries any."""
    extended = _find(placemark, "ExtendedData")
    if extended is None:
        return {}
    fields = {}
    for data in (c for c in extended if _local(c) == "Data"):
        name = data.get("name", "")
        if name.startswith(DATA_PREFIX):
            fields[name[len(DATA_PREFIX) :]] = _text(_find(data, "value"))
    return fields


def _coordinates(text: str) -> list[tuple[float, float, float | None]]:
    return [_triple(item) for item in text.split() if item]


def _triple(item: str) -> tuple[float, float, float | None]:
    parts = item.split(",")
    if len(parts) < 2:
        raise KmlError(f"cannot read the coordinate {item!r}")
    try:
        lon, lat = float(parts[0]), float(parts[1])
        alt = float(parts[2]) if len(parts) > 2 and parts[2] != "" else None
    except ValueError:
        raise KmlError(f"cannot read the coordinate {item!r}") from None
    return lon, lat, alt


def _altitude(alt_m: float | None) -> float | None:
    """A coordinate's altitude as feet, or none: zero in a KML file means
    "clamped to the ground", not sea level."""
    if alt_m is None or alt_m == 0:
        return None
    return round(alt_m * FT_PER_M)


def _float(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _check(waypoint: ImportedWaypoint) -> None:
    for label, value in (("latitude", waypoint.lat), ("longitude", waypoint.lon)):
        if not math.isfinite(value):
            raise KmlError(f"{waypoint.name}: {label} is not a number")
    if abs(waypoint.lat) > MAX_LATITUDE_DEG:
        raise KmlError(f"{waypoint.name}: latitude {waypoint.lat} is beyond {MAX_LATITUDE_DEG}")
    if abs(waypoint.lon) > 180:
        raise KmlError(f"{waypoint.name}: longitude {waypoint.lon} is out of range")
