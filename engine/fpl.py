"""The route as a Garmin flight plan (`.fpl`), and a flight plan back into a route.

FlightPlan v1 (http://www8.garmin.com/xmlschemas/FlightPlanv1.xsd) is a
waypoint table plus a route that names table entries in flying order. It is
what ForeFlight, Garmin Pilot and Garmin panel avionics all import.

Only the pilot's own points go in. TOC and TOD are this planner's answer to
today's weather; ForeFlight and the avionics work out their own, and a user
waypoint pinned where ours fell would only be wrong tomorrow. Nor is there
anywhere to put an altitude or the mission: the format has neither, which is
why the KML stays the file that reloads here.

The schema is strict about characters, and a device that finds one it does
not like refuses the whole file. Identifiers are `[A-Z0-9]{1,12}` and
comments and the route name `[A-Z0-9 /]{0,25}`, so every name is folded into
those before it is written.

Reading one back gives positions and names only: whatever made the file,
there was nowhere in it for a crossing altitude or a leg's climb or descent.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from engine.atmosphere import FT_PER_M
from engine.geo import MAX_LATITUDE_DEG
from engine.kml import ImportedWaypoint, KmlFile, decode_xml
from engine.navlog import Waypoint

__all__ = ["FplError", "flight_plan", "fpl_filename", "looks_like_fpl", "read_fpl"]

FPL_NS = "http://www8.garmin.com/xmlschemas/FlightPlan/v1"

AIRPORT_KINDS = ("airport", "large_airport", "medium_airport", "small_airport")

IDENTIFIER_MAX = 12
COMMENT_MAX = 25
# The schema's own limit on route points.
ROUTE_MAX = 300

_IDENTIFIER = re.compile(rf"^[A-Z0-9]{{1,{IDENTIFIER_MAX}}}$")


class FplError(ValueError):
    """The route cannot be written as a flight plan, or the file is not one."""


def flight_plan(waypoints: list[Waypoint], *, created: datetime | None = None) -> str:
    """The pilot's waypoints as a FlightPlan v1 document.

    An airport goes in as an `AIRPORT` by its identifier, so the importer uses
    its own database entry for it. Anything else is a `USER WAYPOINT` at its
    position, its identifier made from the pilot's name for it. The name
    itself, as near as the character set allows, is every point's comment,
    which is what ForeFlight and the avionics show beside the identifier.
    """
    points = [w for w in waypoints if not w.generated]
    if len(points) < 2:
        raise FplError("the flight plan needs a departure and a destination")
    if len(points) > ROUTE_MAX:
        raise FplError(f"a flight plan holds at most {ROUTE_MAX} waypoints")

    ET.register_namespace("", FPL_NS)
    root = ET.Element(_tag("flight-plan"))
    stamp = (created or datetime.now(UTC)).astimezone(UTC)
    _child(root, "created", stamp.strftime("%Y-%m-%dT%H:%M:%SZ"))
    table = _child(root, "waypoint-table")
    route = _child(root, "route")
    _child(route, "route-name", _text_field(f"{points[0].name} TO {points[-1].name}"))
    _child(route, "flight-plan-index", "1")

    # A route that comes back through the same point names one table entry
    # twice; two different points that fold to the same identifier get a
    # numbered one each.
    assigned: dict[tuple[str, str, tuple[float, float]], str] = {}
    taken: set[str] = set()
    for number, waypoint in enumerate(points, start=1):
        kind = "AIRPORT" if _is_airport(waypoint) else "USER WAYPOINT"
        position = (round(waypoint.position.lat, 6), round(waypoint.position.lon, 6))
        key = (kind, waypoint.name, position)
        identifier = assigned.get(key)
        if identifier is None:
            if kind == "AIRPORT":
                identifier = waypoint.name.strip().upper()
            else:
                identifier = _unique(_identifier(waypoint.name) or f"WP{number}", taken)
            assigned[key] = identifier
            taken.add(identifier)
            entry = _child(table, "waypoint")
            _child(entry, "identifier", identifier)
            _child(entry, "type", kind)
            # Empty is what the schema asks of a user waypoint. An airport's
            # would be its ICAO region (K1..K7 in the US), which the airport
            # database does not carry; importers find the airport by its
            # identifier without it.
            _child(entry, "country-code", "")
            _child(entry, "lat", f"{position[0]:.6f}")
            _child(entry, "lon", f"{position[1]:.6f}")
            _child(entry, "comment", _text_field(waypoint.name))

        point = _child(route, "route-point")
        _child(point, "waypoint-identifier", identifier)
        _child(point, "waypoint-type", kind)
        _child(point, "waypoint-country-code", "")

    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def fpl_filename(waypoints: list[Waypoint]) -> str:
    """`route-KSQL-KSBP.fpl`, or `route.fpl` if the route has no names."""
    names = [w.name for w in waypoints if not w.generated]
    parts = [re.sub(r"[^A-Za-z0-9]+", "", n)[:12] for n in (names[0], names[-1])] if names else []
    return "-".join(["route", *(p for p in parts if p)]) + ".fpl"


# --- import ----------------------------------------------------------------


def looks_like_fpl(data: bytes) -> bool:
    """Whether the bytes are a flight plan rather than KML.

    Read off the root element near the top of the file, so the extension
    the browser happened to keep does not decide which parser runs. The
    bytes are decoded first: in a UTF-16 file every letter is two bytes.
    """
    try:
        head = decode_xml(data[:8192], FplError)
    except FplError:
        return False
    return re.search(r"<(?:[A-Za-z_][\w.-]*:)?flight-plan[\s>/]", head[:4096]) is not None


def read_fpl(data: bytes) -> KmlFile:
    """The route in a flight plan, in flying order, in the shape KML import
    returns.

    The route section names entries in the waypoint table; a file with no
    route is taken as its table in the order written, which the schema does
    not promise is the order flown, so that comes with a warning. An
    `AIRPORT` is marked as one, for the caller to look up. Every other type
    -- user waypoint, VOR, NDB, intersection -- is kept at the position the
    file gives it.
    """
    root = _parse(data)
    table: dict[tuple[str, str, str], ImportedWaypoint] = {}
    in_order: list[ImportedWaypoint] = []
    for node in (n for n in root.iter() if _local(n) == "waypoint"):
        identifier = _field(node, "identifier").upper()
        if not identifier:
            continue
        kind = _field(node, "type").upper()
        point = ImportedWaypoint(
            name=_name(identifier, kind, _field(node, "comment")),
            lat=_coordinate(node, "lat", identifier),
            lon=_coordinate(node, "lon", identifier),
            kind="airport" if kind == "AIRPORT" else "waypoint",
            # ForeFlight's own addition, a crossing altitude per point; empty
            # on every point it has nothing to say about.
            altitude_ft=_number(_field(node, "altitude-ft")),
            elevation_ft=_elevation(_field(node, "elevation")),
            # The file says which of its points are airports, so the caller
            # trusts that rather than guessing from the name.
            from_extended_data=True,
        )
        _check(point)
        table.setdefault((identifier, kind, _field(node, "country-code").upper()), point)
        in_order.append(point)

    warnings: list[str] = []
    waypoints: list[ImportedWaypoint] = []
    route_points = [n for n in root.iter() if _local(n) == "route-point"]
    for node in route_points:
        identifier = _field(node, "waypoint-identifier").upper()
        kind = _field(node, "waypoint-type").upper()
        country = _field(node, "waypoint-country-code").upper()
        point = table.get((identifier, kind, country)) or next(
            (p for (i, k, _c), p in table.items() if i == identifier and k == kind),
            None,
        )
        if point is None:
            warnings.append(
                f"{identifier} is in the route but not in the waypoint table; left out."
            )
            continue
        waypoints.append(point)
    if not route_points and in_order:
        waypoints = in_order
        warnings.append(
            "The file has no route, only waypoints; they are listed in the order written."
        )
    if not waypoints:
        raise FplError("no waypoints found in the flight plan")
    return KmlFile(waypoints=waypoints, warnings=warnings)


def _parse(data: bytes) -> ET.Element:
    text = decode_xml(data, FplError)
    # Refused rather than parsed with entities disabled, as for KML: expat's
    # entity handling cannot be turned off from ElementTree.
    if re.search(r"<!\s*DOCTYPE", text, re.IGNORECASE):
        raise FplError("file declares a document type, which a flight plan never does")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise FplError(f"not well-formed XML: {exc}") from None
    if _local(root) != "flight-plan":
        raise FplError("not a Garmin flight plan")
    return root


def _name(identifier: str, kind: str, comment: str) -> str:
    """The comment, when the identifier was evidently made from it -- a label
    folded to fit, as this program's own export does -- and the identifier
    otherwise. A comment is free text on most devices, often a date."""
    comment = re.sub(r"\s+", " ", comment).strip()
    if kind != "AIRPORT" and comment and _identifier(comment) == identifier:
        return comment
    return identifier


def _local(node: ET.Element) -> str:
    """The tag without its namespace; some writers leave Garmin's off."""
    return node.tag.rsplit("}", 1)[-1]


def _field(node: ET.Element, name: str) -> str:
    child = next((c for c in node if _local(c) == name), None)
    return (child.text or "").strip() if child is not None else ""


def _coordinate(node: ET.Element, name: str, identifier: str) -> float:
    text = _field(node, name)
    try:
        return float(text)
    except ValueError:
        raise FplError(f"{identifier}: cannot read the {name} {text!r}") from None


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _elevation(text: str) -> float | None:
    """The schema's elevation, which is in metres, as feet."""
    try:
        metres = float(text)
    except ValueError:
        return None
    return round(metres * FT_PER_M) if math.isfinite(metres) else None


def _check(point: ImportedWaypoint) -> None:
    for label, value in (("latitude", point.lat), ("longitude", point.lon)):
        if not math.isfinite(value):
            raise FplError(f"{point.name}: {label} is not a number")
    if abs(point.lat) > MAX_LATITUDE_DEG:
        raise FplError(f"{point.name}: latitude {point.lat} is beyond {MAX_LATITUDE_DEG}")
    if abs(point.lon) > 180:
        raise FplError(f"{point.name}: longitude {point.lon} is out of range")


def _is_airport(waypoint: Waypoint) -> bool:
    # An airport whose name is no longer an identifier -- renamed, or never
    # one -- cannot be looked up, so it goes in at its position instead.
    return waypoint.kind in AIRPORT_KINDS and bool(_IDENTIFIER.match(waypoint.name.strip().upper()))


def _identifier(name: str) -> str:
    """The name upper-cased, without what an identifier may not hold."""
    return re.sub(r"[^A-Z0-9]", "", name.upper())[:IDENTIFIER_MAX]


def _unique(identifier: str, taken: set[str]) -> str:
    """The identifier, or the first numbered variant of it not yet used."""
    if identifier not in taken:
        return identifier
    for n in range(2, 10_000):
        suffix = str(n)
        candidate = identifier[: IDENTIFIER_MAX - len(suffix)] + suffix
        if candidate not in taken:
            return candidate
    raise FplError(f"too many waypoints named {identifier}")


def _text_field(text: str) -> str:
    """A comment or route name: upper case, spaces and slashes, 25 long."""
    folded = re.sub(r"[^A-Z0-9 /]", " ", text.upper())
    return re.sub(r" +", " ", folded).strip()[:COMMENT_MAX].rstrip()


def _tag(name: str) -> str:
    return f"{{{FPL_NS}}}{name}"


def _child(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    node = ET.SubElement(parent, _tag(tag))
    if text is not None:
        node.text = text
    return node
