"""Geodesy and the wind triangle.

Distances and bearings on a **spherical** earth, plus the wind-triangle
solution a navlog needs to turn a course into a heading.
since we make a spherical assumption - reject polar destinations

Great circle formula:
Useful if the distance between legs are < 200 n.m.

Angles are degrees, distances nautical miles, speeds knots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

M_PER_NM = 1852.0  # exact, by definition
EARTH_RADIUS_NM = 6371008.8 / M_PER_NM  # IUGG mean radius

# Beyond this the spherical model is refused
MAX_LATITUDE_DEG = 80.0

class GeodesicError(ValueError):
    """A geodesic could not be computed."""

class PolarRegionUnsupported(GeodesicError):
    """A position lies too near a pole for the spherical model to be useful."""

class WindTooStrong(ValueError):
    """The crosswind exceeds true airspeed, so the course cannot be held."""

@dataclass(frozen=True)
class LatLon:
    """A geographic position in decimal degrees, north and east positive."""

    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"latitude {self.lat:g} is outside -90 to 90 degrees")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"longitude {self.lon:g} is outside -180 to 180 degrees")


@dataclass(frozen=True)
class Segment:
    """A great-circle path between two points, with its measurements.
    """

    start: LatLon
    end: LatLon
    distance_nm: float
    true_course_deg: float

    def point_at_nm(self, along_nm: float) -> LatLon:
        """The point a given distance along the track from `start`.
       """
        return direct(self.start, self.true_course_deg, along_nm)

    def point_at_fraction(self, fraction: float) -> LatLon:
        """The point a given fraction of the way along the segment."""
        return self.point_at_nm(self.distance_nm * fraction)

    def cross_track_nm(self, point: LatLon) -> float:
        """Signed perpendicular distance from `point` to this track.

        Positive means the point lies right of the track. Used to build the
        route corridor and to measure how far a candidate waypoint deviates.

        Spherical rather than ellipsoidal: the error is a few metres over a
        VFR leg, far below the tens of miles a corridor is measured in.
        """
        leg = inverse(self.start, point)
        if leg.distance_nm == 0.0:
            return 0.0
        relative = math.radians(leg.true_course_deg - self.true_course_deg)
        angular = leg.distance_nm / EARTH_RADIUS_NM
        return math.asin(math.sin(angular) * math.sin(relative)) * EARTH_RADIUS_NM

    def along_track_nm(self, point: LatLon) -> float:
        """How far along this track the closest point to `point` lies.

        Negative if the point falls before `start`; may exceed `distance_nm`
        if it falls beyond `end`. Together with `cross_track_nm` this decides
        whether a candidate waypoint is inside the corridor.
        """
        leg = inverse(self.start, point)
        if leg.distance_nm == 0.0:
            return 0.0
        angular = leg.distance_nm / EARTH_RADIUS_NM
        cross = self.cross_track_nm(point) / EARTH_RADIUS_NM
        ratio = math.cos(angular) / math.cos(cross)
        along = math.acos(max(-1.0, min(1.0, ratio))) * EARTH_RADIUS_NM
        # acos loses the sign, so recover it from the course difference.
        delta = bearing_difference(self.true_course_deg, leg.true_course_deg)
        return -along if abs(delta) > 90.0 else along


@dataclass(frozen=True)
class WindTriangle:
    """The solution of course, wind and true airspeed for a leg.

    Course and heading are both carried because the pair is what a pilot
    works with -- the course is what was asked for, the heading is what gets
    flown. The crab angle between them is derived rather than stored, so the
    three can never disagree.
    """

    true_course_deg: float  # what the leg asked for
    true_heading_deg: float  # what must be flown to hold it
    ground_speed_kt: float
    headwind_kt: float  # negative for a tailwind
    crosswind_kt: float  # positive from the right

    @property
    def wind_correction_angle_deg(self) -> float:
        """Signed crab, positive when the heading is right of the course."""
        return wrap_angle_deg(self.true_heading_deg - self.true_course_deg)


def _require_usable_latitude(point: LatLon, role: str) -> None:
    if abs(point.lat) > MAX_LATITUDE_DEG:
        raise PolarRegionUnsupported(
            f"{role} at latitude {point.lat:g} is beyond the supported "
            f"{MAX_LATITUDE_DEG:g} degrees; near the poles a single true course "
            f"does not describe the path and the spherical model is unreliable"
        )
# Angle utils
def wrap_angle_deg(angle:float) -> float:
    """
    Wraps an angle (deg) to [-180.0, 180.0)
    """
    return (angle + 540.0) % 360.0 - 180.0

def normalise_bearing(degrees: float) -> float:
    """Wrap an angle into [0, 360)."""
    return degrees % 360.0

def bearing_difference(first_deg: float, second_deg: float) -> float:
    """Signed smallest turn from one bearing to another, in [-180, 180).

    An exact reversal is reported as -180 rather than +180. Either is equally
    correct -- the turn is the same size in both directions -- and callers
    (the router's turn penalty, for one) use the magnitude, so the ambiguity
    is left alone rather than special-cased.
    """
    return (second_deg - first_deg + 180.0) % 360.0 - 180.0

# --- spherical geodesics -------------------------------------------------
def inverse(start: LatLon, end: LatLon) -> Segment:
    """The segment between two points: its distance and its true course."""
    _require_usable_latitude(start, "start point")
    _require_usable_latitude(end, "end point")

    if start.lat == end.lat and start.lon == end.lon:
        return Segment(start=start, end=end, distance_nm=0.0, true_course_deg=0.0)

    delta_lat = math.radians(end.lat - start.lat)
    delta_lon = math.radians(end.lon - start.lon)
    a = (
        math.sin(0.5 * delta_lat) ** 2
        + math.cos(math.radians(start.lat))
        * math.cos(math.radians(end.lat))
        * math.sin(0.5 * delta_lon) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return Segment(
        start=start,
        end=end,
        distance_nm=EARTH_RADIUS_NM * c,
        true_course_deg=_bearing(start, end),
    )

def _bearing(start: LatLon, end: LatLon) -> float:
    """Initial great-circle bearing from one point to another, in degrees."""
    lat1, lat2 = math.radians(start.lat), math.radians(end.lat)
    delta_lon = math.radians(end.lon - start.lon)
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        delta_lon
    )
    return math.degrees(math.atan2(y, x)) % 360.0


def direct(start: LatLon, bearing_deg: float, distance_nm: float) -> LatLon:
    """The point reached by flying distance_nm along bearing_deg from start
    """
    _require_usable_latitude(start, "start point")
    if distance_nm == 0.0:
        return start

    angular = distance_nm / EARTH_RADIUS_NM
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(start.lat)
    lon1 = math.radians(start.lon)

    sin_lat2 = math.sin(lat1) * math.cos(angular) + math.cos(lat1) * math.sin(
        angular) * math.cos(bearing)
    if(sin_lat2 > 1.0 or sin_lat2 < -1.0):
        raise GeodesicError(
            f"sin of the resulting latitude is {sin_lat2!r}, outside [-1, 1]; "
            f"flying {distance_nm:g} nm on {bearing_deg:g} from {start}"
        )
    lat2 = math.asin(sin_lat2)
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * sin_lat2,
    )

    # Wrap longitude to [-180.0, 180.0)
    end = LatLon(math.degrees(lat2), wrap_angle_deg(math.degrees(lon2)))
    _require_usable_latitude(end, "resulting point")
    return end


def distance_nm(start: LatLon, end: LatLon) -> float:
    """Shorthand for the distance alone."""
    return inverse(start, end).distance_nm


def true_course_deg(start: LatLon, end: LatLon) -> float:
    """Shorthand for the true course alone."""
    return inverse(start, end).true_course_deg


# --- the wind triangle ---------------------------------------------------

def solve_wind_triangle(
    true_course_deg: float,
    true_airspeed_kt: float,
    wind_from_deg: float,
    wind_speed_kt: float,
) -> WindTriangle:
    """Solve heading and ground speed for a desired course.
    wca = TH - TC
    Raises `WindTooStrong` when the crosswind component exceeds true airspeed
    and no heading can hold the course -- a real answer, not a failure.
    """
    if true_airspeed_kt <= 0.0:
        raise ValueError("true airspeed must be positive")

    # Where the wind is *going*, relative to the course
    towards = math.radians(wrap_angle_deg(wind_from_deg + 180.0 - true_course_deg))
    # The crab is into the wind, so a wind from the right means a heading right of
    # course, and the two signs must agree.
    crosswind = -wind_speed_kt * math.sin(towards)
    sin_wca = crosswind / true_airspeed_kt
    if abs(sin_wca) > 1.0:
        raise WindTooStrong(
            f"crosswind of {abs(crosswind):.1f} kt exceeds true airspeed of "
            f"{true_airspeed_kt:.1f} kt; the course cannot be held"
        )

    wca = math.asin(sin_wca)
    # Along-course component, headwind would be negative here
    wind_along_course = wind_speed_kt * math.cos(towards)
    ground_speed = true_airspeed_kt * math.cos(wca) + wind_along_course
    return WindTriangle(
        true_course_deg=true_course_deg,
        true_heading_deg=normalise_bearing(true_course_deg + math.degrees(wca)),
        ground_speed_kt=ground_speed,
        headwind_kt=-wind_along_course, # Sign flip so positive reads a headwind
        crosswind_kt=crosswind,
    )
