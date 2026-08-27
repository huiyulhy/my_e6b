"""Magnetic variation from the NOAA World Magnetic Model (WMM2025).

Used to determine the magnetic declination for a given location
MC = TC - var, and +ve var = east -- "east is least,
west is best".  

This is a direct implementation of WMM2025, valid 2025.0 to 2030.0, following
the technical report's equations and notation: (7)-(8) coordinates, (9) secular
drift, (10)-(12) field components, (16) Legendre derivative.
WMM coefficients source: https://www.ncei.noaa.gov/products/world-magnetic-model
Other references: https://github.com/boxpet/pygeomag/blob/main/pygeomag/geomag.py
full technical report: https://repository.library.noaa.gov/view/noaa/71569
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

COF_PATH = Path(__file__).resolve().parent.parent / "data" / "magnetic" / "WMM2025.COF"

MAX_DEGREE = 12

# WGS-84 reference ellipsoid.
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Geomagnetic reference radius used by the model. Not the ellipsoid radius.
GEOMAG_R_KM = 6371.2

KM_PER_FT = 0.0003048

class OutsideModelValidity(ValueError):
    """A date fell outside the model's five-year validity window."""


class AtGeographicPole(ValueError):
    """A position sat on a pole, where the report's equations are singular."""


# cos(phi') below this and the 1/cos(phi') in eq. (11) -- and the sec(phi') in
# eq. (16) -- have lost every significant figure. s
MIN_COS_PHI_PRIME = 1e-10

@dataclass(frozen=True)
class MagneticField:
    """The geomagnetic field at a point, in the local geodetic frame."""

    declination_deg: float  # variation; positive east
    inclination_deg: float  # dip angle; positive down
    north_nt: float
    east_nt: float
    down_nt: float

    @property
    def horizontal_nt(self) -> float:
        return math.hypot(self.north_nt, self.east_nt)

    @property
    def total_nt(self) -> float:
        return math.hypot(self.horizontal_nt, self.down_nt)


@dataclass(frozen=True)
class _Model:
    epoch: float
    label: str
    # g[n][m], h[n][m] and their annual rates of change.
    g: list[list[float]]
    h: list[list[float]]
    dg: list[list[float]]
    dh: list[list[float]]


@lru_cache(maxsize=2)
def _model(path: Path | None = None) -> _Model:
    """Parse a WMM .COF coefficient file.

    Format is one header line giving the epoch, then one line per coefficient:
    degree (n), order (m), g, h, dg/dt, dh/dt. The file ends with lines of 9s.
    """
    size = MAX_DEGREE + 1
    g = [[0.0] * size for _ in range(size)]
    h = [[0.0] * size for _ in range(size)]
    dg = [[0.0] * size for _ in range(size)]
    dh = [[0.0] * size for _ in range(size)]

    lines = (path or COF_PATH).read_text().splitlines()
    header = lines[0].split()
    epoch, label = float(header[0]), header[1]

    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 6 or line.lstrip().startswith("9999"):
            continue
        n, m = int(parts[0]), int(parts[1])
        if n > MAX_DEGREE:
            continue
        g[n][m], h[n][m], dg[n][m], dh[n][m] = (float(v) for v in parts[2:])

    return _Model(epoch=epoch, label=label, g=g, h=h, dg=dg, dh=dh)


def _legendre_in_phi(phi_prime_rad: float) -> list[list[float]]:
    """Schmidt semi-normalised degree and order up to 13.

    These are Legendre *functions*, for odd m they carry a
    factor cosᵐφ′, which is where the sectoral seed below gets its `cos_phi`.

    Textbooks state this recursion in colatitude, where sin(phi') = cos(theta)
    and cos(phi') = sin(theta): `sin_phi` is the usual argument and `cos_phi`
    the usual sin(theta)**n seed factor. 
    """
    size = MAX_DEGREE + 2
    p = [[0.0] * size for _ in range(size)]
    sin_phi, cos_phi = math.sin(phi_prime_rad), math.cos(phi_prime_rad)

    p[0][0] = 1.0
    for n in range(1, size):
        factor = 1.0 if n == 1 else math.sqrt((2 * n - 1) / (2 * n))
        p[n][n] = factor * cos_phi * p[n - 1][n - 1]
        for m in range(n):
            older = p[n - 2][m] if n - 2 >= m else 0.0
            a = math.sqrt((n - 1) ** 2 - m * m)
            b = math.sqrt(n * n - m * m)
            p[n][m] = ((2 * n - 1) * sin_phi * p[n - 1][m] - a * older) / b

    return p


def _d_legendre_in_phi(
    p: list[list[float]], phi_prime_rad: float
) -> list[list[float]]:
    """
    dP/dPhi
    """
    size = MAX_DEGREE + 1
    dp = [[0.0] * size for _ in range(size)]
    tan_phi = math.tan(phi_prime_rad)
    sec_phi = 1.0 / math.cos(phi_prime_rad)

    for n in range(1, size):
        for m in range(n + 1):
            dp[n][m] = (n + 1) * tan_phi * p[n][m] - math.sqrt(
                (n + 1) ** 2 - m * m
            ) * sec_phi * p[n + 1][m]

    return dp


def field(
    latitude_deg: float,
    longitude_deg: float,
    altitude_ft: float = 0.0,
    decimal_year: float | None = None,
    *,
    cof_path: Path | None = None,
) -> MagneticField:
    """Evaluate the geomagnetic field at a point and time.

    @param decimal_year: fractional calendar year; `decimal_year_for` builds
    one from a date, and None means today.
    Outside the model's five-year window this raises rather
    than extrapolating -- see `check_validity`.
    """
    if decimal_year is None:
        decimal_year = decimal_year_now()
    model = _model(cof_path)
    check_validity(decimal_year, cof_path=cof_path)

    dt = decimal_year - model.epoch

    lat_rad = math.radians(latitude_deg)
    lon_rad = math.radians(longitude_deg)  # lambda
    height_km = altitude_ft * KM_PER_FT

    # Convert to geocentric spherical coordinates
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    rc = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    p_axis = (rc + height_km) * cos_lat
    z_axis = (rc * (1.0 - WGS84_E2) + height_km) * sin_lat
    radius_km = math.hypot(p_axis, z_axis)
    phi_prime = math.asin(z_axis / radius_km)

    cos_phi_prime = math.cos(phi_prime)
    if abs(cos_phi_prime) < MIN_COS_PHI_PRIME:
        raise AtGeographicPole(
            f"latitude {latitude_deg:g} deg is on a geographic pole, singularity in cos_phi_prime "
        )

    # Get legendre coefficients
    legendre = _legendre_in_phi(phi_prime)
    d_legendre = _d_legendre_in_phi(legendre, phi_prime)

    # Eqs. (10)-(12), summed together since they share the inner terms.
    x_prime = y_prime = z_prime = 0.0
    ratio = GEOMAG_R_KM / radius_km
    for n in range(1, MAX_DEGREE + 1):
        scale = ratio ** (n + 2)
        for m in range(n + 1):
            g = model.g[n][m] + dt * model.dg[n][m]
            h = model.h[n][m] + dt * model.dh[n][m]
            cos_ml = math.cos(m * lon_rad)
            sin_ml = math.sin(m * lon_rad)
            x_prime -= scale * (g * cos_ml + h * sin_ml) * d_legendre[n][m]
            y_prime += scale * m * (g * sin_ml - h * cos_ml) * legendre[n][m]
            z_prime -= scale * (n + 1) * (g * cos_ml + h * sin_ml) * legendre[n][m]
    y_prime /= cos_phi_prime  # the 1/cos(phi') outside eq. (11)'s sum

    # Rotate the geocentric components into the local geodetic frame.
    delta = phi_prime - lat_rad
    cos_d, sin_d = math.cos(delta), math.sin(delta)
    north = x_prime * cos_d - z_prime * sin_d
    down = x_prime * sin_d + z_prime * cos_d
    east = y_prime

    horizontal = math.hypot(north, east)
    return MagneticField(
        declination_deg=math.degrees(math.atan2(east, north)),
        inclination_deg=math.degrees(math.atan2(down, horizontal)),
        north_nt=north,
        east_nt=east,
        down_nt=down,
    )


def variation(
    latitude_deg: float,
    longitude_deg: float,
    altitude_ft: float = 0.0,
    decimal_year: float | None = None,
    **kwargs,
) -> float:
    """Magnetic variation in degrees, positive east.

    The only thing a navlog actually needs from this module. `decimal_year`
    defaults to today.
    """
    return field(latitude_deg, longitude_deg, altitude_ft, decimal_year, **kwargs).declination_deg


def validity_window(cof_path: Path | None = None) -> tuple[float, float]:
    """The decimal-year range the loaded coefficients are valid over."""
    model = _model(cof_path)
    return model.epoch, model.epoch + 5.0


def check_validity(decimal_year: float, *, cof_path: Path | None = None) -> None:
    """Raise `OutsideModelValidity` if a date falls outside the model window.s
    """
    model = _model(cof_path)
    start, end = validity_window(cof_path)
    if not (start <= decimal_year <= end):
        raise OutsideModelValidity(
            f"{model.label} is valid from {start:g} to {end:g}; asked for "
            f"{decimal_year:g}. The model has expired for that date -- download "
            f"the current coefficients from NOAA and replace "
            f"{COF_PATH.name}."
        )


def decimal_year_for(year: int, month: int = 1, day: int = 1) -> float:
    """Convert a calendar date to the fractional year the model expects."""
    date = datetime.date(year, month, day)
    start = datetime.date(year, 1, 1)
    days_in_year = (datetime.date(year + 1, 1, 1) - start).days
    return year + (date - start).days / days_in_year


def decimal_year_now() -> float:
    """Today, as a fractional year.
s
    """
    today = datetime.date.today()
    return decimal_year_for(today.year, today.month, today.day)


def true_to_magnetic(true_deg: float, variation_deg: float) -> float:
    """Convert a true course or heading to magnetic.
    MC/MH = TC/TH - var (+ve = east)
    """
    return (true_deg - variation_deg) % 360.0


def magnetic_to_true(magnetic_deg: float, variation_deg: float) -> float:
    """Convert a magnetic course or heading to true."""
    return (magnetic_deg + variation_deg) % 360.0
