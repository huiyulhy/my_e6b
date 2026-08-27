"""International Standard Atmosphere and altitude conversions.

    quantity              ISA (used here) 
    tropopause pressure   22632.0 Pa
Calculations use SI units (for distance, temperature, pressure and speed)
Note that we use the ISA model here, which is slightly different than the US std atmos model
References:
1. https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf
2. https://agodemar.github.io/FlightMechanics4Pilots/mypages/international-standard-atmosphere/
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


class AboveModelCeiling(ValueError):
    """A query fell above the troposphere, where this module does not model.
    """

# ISA Constants
T0_C = 15.0  # sea level standard temperature, degC
T0_K = T0_C + 273.15  # 288.15 K
P0_PA = 101325.0  # sea level standard pressure, Pa
P0_INHG = 29.92126  # sea level standard pressure, inHg
G0 = 9.80665  # standard gravity, m/s^2
R_AIR = 287.05287  # specific gas constant for dry air, J/(kg K)
LAPSE_K_PER_M = -0.0065  # troposphere lapse rate, K/m (exactly 6.5 K/km) (lambda)
TROPOPAUSE_M = 11000.0  # base of the isothermal layer, m
RHO0 = 1.2250  # kg/m^3 , ICAO's published sea level density

# Derived rather than hardcoded, so that changing T0 or the lapse rate carries
# through. Pinning this separately is how a model drifts out of agreement with
# itself.
T_TROPOPAUSE_K = T0_K + LAPSE_K_PER_M * TROPOPAUSE_M  # 216.65 K = -56.50 degC

EARTH_RADIUS_M = 6356766.0  # polar radius, for geopotential conversions (R)

# The model's ceiling, in the units callers work in. 36089 ft.
TROPOPAUSE_CEILING_FT = TROPOPAUSE_M * 3.280839895013123

# --- unit conversions ----------------------------------------------------

FT_PER_M = 3.280839895013123
M_PER_FT = 1.0 / FT_PER_M
PA_PER_INHG = P0_PA / P0_INHG
KTS_TO_MPS = 1852.0 / 3600.0  # a knot is one NM (1852 m, exact) per hour
MPS_TO_KTS = 1.0 / KTS_TO_MPS

# Exponent for the troposphere pressure relation, P/P0 = (T/T0)^(-g0/(L R)).
_DELTA_EXP = -G0 / (LAPSE_K_PER_M * R_AIR)  # Exponent for pressure ratio
_SIGMA_EXP = _DELTA_EXP - 1                 # Exponent for density ratio

def ft_to_m(ft: float) -> float:
    return ft * M_PER_FT

def m_to_ft(m: float) -> float:
    return m * FT_PER_M

def c_to_k(c: float) -> float:
    return c + 273.15

def k_to_c(k: float) -> float:
    return k - 273.15

def inhg_to_pa(inhg: float) -> float:
    return inhg * PA_PER_INHG

def pa_to_inhg(pa: float) -> float:
    return pa / PA_PER_INHG

# --- standard atmosphere -------------------------------------------------

def isa_temperature_c(altitude_ft: float) -> float:
    """Standard-atmosphere temperature at a geopotential altitude, in Celsius.
    Only use the model up to 11000m (we don't go above)
    """
    h_m = ft_to_m(altitude_ft)
    if h_m > TROPOPAUSE_M:
        raise AboveModelCeiling(
            f"{altitude_ft:g} ft is above the tropopause ({TROPOPAUSE_CEILING_FT:.0f} ft); "
            f"this model covers the troposphere only"
        )
    return k_to_c(T0_K + LAPSE_K_PER_M * h_m)


def isa_pressure_pa(altitude_ft: float) -> float:
    """Standard-atmosphere pressure at a geopotential altitude, in pascals.
    Calculates the pressure at a given altitude, assuming isa at sea level
    Only use the model up to 11000m (we don't go above)
    """
    h_m = ft_to_m(altitude_ft)
    if h_m > TROPOPAUSE_M:
        raise AboveModelCeiling(
            f"{altitude_ft:g} ft is above the tropopause ({TROPOPAUSE_CEILING_FT:.0f} ft); "
            f"this model covers the troposphere only"
        )
    t_k = T0_K + LAPSE_K_PER_M * h_m
    return P0_PA * (t_k / T0_K) ** _DELTA_EXP


def isa_pressure_inhg(altitude_ft: float) -> float:
    return pa_to_inhg(isa_pressure_pa(altitude_ft))


def isa_density(altitude_ft: float) -> float:
    """Standard-atmosphere density at a geopotential altitude, in kg/m^3."""
    t_k = c_to_k(isa_temperature_c(altitude_ft))
    return isa_pressure_pa(altitude_ft) / (R_AIR * t_k)


def isa_deviation_c(altitude_ft: float, oat_c: float) -> float:
    """
    oat temperature above the ISA temperature for a given altitude (in ft)
    """
    return oat_c - isa_temperature_c(altitude_ft)


# --- relative ratios -----------------------------------------------------
#
# The ISA-referenced ratios delta, Theta, sigma. - Note that the base 
# relation of 1 quantity has to be at sea level for the relation to be valid

def relative_pressure(pressure_altitude_ft: float) -> float:
    """delta = p / p_SL, from the pressure altitude alone.

    Exact rather than assumed: pressure altitude is defined as the standard
    altitude at which the ambient pressure occurs, so applying the standard
    pressure law to it recovers the actual pressure whatever the temperature.
    """
    return isa_pressure_pa(pressure_altitude_ft) / P0_PA


def relative_temperature(oat_c: float) -> float:
    """Theta = T / T_SL, from the measured outside air temperature.

    Note that this takes an OAT, not an altitude. Theta is the one ratio the
    standard atmosphere cannot supply for a real day; it has to be observed.
    """
    return c_to_k(oat_c) / T0_K


def relative_density(density_altitude_ft: float) -> float:
    """sigma = rho / rho_SL, from the density altitude alone.
    Gives us the ratio of density to sea level given the density altitude
    """
    t_ratio = 1.0 + LAPSE_K_PER_M * ft_to_m(density_altitude_ft) / T0_K
    return t_ratio**_SIGMA_EXP


def altitude_for_pressure_pa(pressure_pa: float) -> float:
    """Inverse of `isa_pressure_pa`: the standard altitude for a pressure.

    Used to convert a measured or derived pressure back into an altitude,
    which is what pressure altitude and density altitude both are.
    """
    p_trop = P0_PA * (T_TROPOPAUSE_K / T0_K) ** _DELTA_EXP
    if pressure_pa < p_trop:
        raise AboveModelCeiling(
            f"{pressure_pa:.0f} Pa is below the tropopause pressure ({p_trop:.0f} Pa), "
            f"so the altitude is above {TROPOPAUSE_CEILING_FT:.0f} ft; "
            f"this model covers the troposphere only"
        )
    h_m = (T0_K / LAPSE_K_PER_M) * ((pressure_pa / P0_PA) ** (1.0 / _DELTA_EXP) - 1.0)
    return m_to_ft(h_m)


# --- pressure and density altitude ---------------------------------------


def pressure_altitude(field_elevation_ft: float, altimeter_inhg: float) -> float:
    """Pressure altitude: the altitude the altimeter shows when set to 29.92.
    1. Calculate the expected altitude given station pressure - that's the baseline
    2. Add field elevation from station altitude, which brings us to our current altitude
    """
    pa = field_elevation_ft + altitude_for_pressure_pa(inhg_to_pa(altimeter_inhg))
    if ft_to_m(pa) > TROPOPAUSE_M:
        raise AboveModelCeiling(
            f"a pressure altitude of {pa:.0f} ft is above the tropopause "
            f"({TROPOPAUSE_CEILING_FT:.0f} ft); this model covers the troposphere only"
        )
    return pa


def pressure_altitude_approx(field_elevation_ft: float, altimeter_inhg: float) -> float:
    """The cockpit rule of thumb: add 1000 ft for every inHg below 29.92.

    Provided for comparison against `pressure_altitude`; the two agree to
    within a few tens of feet at normal settings and low elevations.
    """
    return field_elevation_ft + (P0_INHG - altimeter_inhg) * 1000.0


def density_altitude(pressure_altitude_ft: float, oat_c: float) -> float:
    """Density altitude: the standard altitude with the same air density.

    delta comes from the pressure altitude, Theta from the OAT, sigma from the
    two, and only the final inversion appeals to the standard atmosphere. This
    is the value performance charts are really keyed to.
    """
    sigma = relative_pressure(pressure_altitude_ft) / relative_temperature(oat_c)
    return altitude_for_relative_density(sigma)


def density_altitude_approx(pressure_altitude_ft: float, oat_c: float) -> float:
    """The rule of thumb: 120 ft of density altitude per degree above ISA."""
    return pressure_altitude_ft + 120.0 * isa_deviation_c(pressure_altitude_ft, oat_c)


def oat_for_density_altitude(
    pressure_altitude_ft: float, density_altitude_ft: float
) -> float:
    """The temperature that puts this pressure altitude at that density altitude.

    The inverse of `density_altitude` in its second argument, and exact rather
    than solved for: sigma comes from the density altitude, delta from the
    pressure altitude, and Theta = delta / sigma is the whole of it.

    Used to move an operating point onto a chart page it is not published on
    without changing the air it describes -- see `performance.cruise_at_density`.
    """
    sigma = relative_density(density_altitude_ft)
    theta = relative_pressure(pressure_altitude_ft) / sigma
    return k_to_c(theta * T0_K)


def altitude_for_relative_density(sigma: float) -> float:
    """Inverse of the sigma relation: the standard altitude for a density ratio.
    This relationship is only valid if the baseline is ISA
    """
    sigma_trop = relative_density(m_to_ft(TROPOPAUSE_M))
    if sigma < sigma_trop:
        raise AboveModelCeiling(
            f"a density ratio of {sigma:.4f} is thinner than the tropopause "
            f"({sigma_trop:.4f}), so the density altitude is above "
            f"{TROPOPAUSE_CEILING_FT:.0f} ft; this model covers the troposphere only"
        )
    t_ratio = sigma ** (1.0 / _SIGMA_EXP)
    h_m = (t_ratio - 1.0) * T0_K / LAPSE_K_PER_M
    return m_to_ft(h_m)


# --- observed temperature ------------------------------------------------
#
# A real day is not the standard atmosphere, and the pilot learns about it from
# two products that look nothing alike: a METAR gives a temperature at a field
# elevation with that station's altimeter setting, an FD winds-aloft forecast
# gives a temperature at a flight altitude. Both are the same thing underneath
# -- one point on a temperature-against-pressure curve -- and this is where
# they are made to agree.
#
# The common currency is ISA deviation against *pressure* altitude. Pressure
# altitude carries no temperature term (see `pressure_altitude`), so the chain
# altitude -> PA -> deviation -> OAT -> density altitude runs one way with
# nothing to solve for. Referencing deviation to PA rather than to indicated
# altitude also means two stations reporting different altimeter settings still
# land in the same coordinate, which is the whole point of reconciling them.

# The dry adiabatic lapse rate: 9.8 K/km, the steepest gradient still air can
# hold without overturning.
DRY_ADIABATIC_C_PER_1000FT = 2.98

# What actually earns a warning. Not the adiabatic rate itself: a shallow
# superadiabatic layer over ground baking in the afternoon sun is ordinary
# weather, and a METAR paired with a forecast nine thousand feet above it will
# cross 3 degC per 1000 ft on any hot day. Only a gradient well past that says
# a digit went in wrong. An inversion is the other way round and is perfectly
# ordinary too, so only this side is worth a word at all.
_IMPLAUSIBLE_LAPSE_C_PER_1000FT = 4.5

# Two observations closer together than this in pressure altitude describe the
# same slice of air, not a gradient. Two airports a couple of hundred feet
# apart with genuinely different weather -- one coastal, one inland -- would
# otherwise imply a lapse rate of tens of degrees per thousand feet and poison
# every altitude above them.
_MERGE_TOLERANCE_FT = 250.0


@dataclass(frozen=True)
class TemperatureSample:
    """One observed point on the temperature curve, in ISA-deviation terms."""

    pressure_altitude_ft: float
    isa_deviation_c: float

    @classmethod
    def observed(
        cls, altitude_ft: float, oat_c: float, altimeter_inhg: float = P0_INHG
    ) -> TemperatureSample:
        """A sample from what was actually read: an altitude and a temperature.

        `altimeter_inhg` is the setting *that observation* was made under --
        the reporting station's for a METAR, the route's for a forecast aloft.
        """
        pa = pressure_altitude(altitude_ft, altimeter_inhg)
        return cls(pa, oat_c - isa_temperature_c(pa))

    @property
    def oat_c(self) -> float:
        return isa_temperature_c(self.pressure_altitude_ft) + self.isa_deviation_c


@dataclass(frozen=True)
class TemperatureProfile:
    """Observed ISA deviation against pressure altitude, linearly interpolated.

    Outside the observed range the nearest deviation is held flat. That is the
    reason the whole model is expressed in deviation and not in temperature:
    holding a *deviation* constant keeps the standard lapse rate running above
    the top sample and below the bottom one, where holding a *temperature*
    constant would claim the air stops cooling with height.

    With no samples at all the profile is a single constant deviation, which is
    exactly the route-wide ISA-deviation figure this replaced -- so a plan with
    no weather entered behaves as it always did.
    """

    samples: tuple[TemperatureSample, ...] = ()
    default_deviation_c: float = 0.0

    @classmethod
    def from_observations(
        cls,
        observations: Iterable[TemperatureSample],
        *,
        default_deviation_c: float = 0.0,
        merge_tolerance_ft: float = _MERGE_TOLERANCE_FT,
    ) -> TemperatureProfile:
        """Sort, merge near-coincident observations, and keep the result.

        Samples win outright where there are any: the default deviation is a
        fallback for an empty profile, never something blended in. A pilot who
        has typed a temperature should see that temperature used, not an
        average of it and a form default they forgot about.
        """
        ordered = sorted(observations, key=lambda s: s.pressure_altitude_ft)
        merged: list[TemperatureSample] = []
        group: list[TemperatureSample] = []

        def flush() -> None:
            if not group:
                return
            merged.append(
                TemperatureSample(
                    sum(s.pressure_altitude_ft for s in group) / len(group),
                    sum(s.isa_deviation_c for s in group) / len(group),
                )
            )

        for sample in ordered:
            if group and (
                sample.pressure_altitude_ft - group[0].pressure_altitude_ft
                > merge_tolerance_ft
            ):
                flush()
                group = []
            group.append(sample)
        flush()

        return cls(tuple(merged), default_deviation_c)

    def deviation_at(self, pressure_altitude_ft: float) -> float:
        if not self.samples:
            return self.default_deviation_c
        if pressure_altitude_ft <= self.samples[0].pressure_altitude_ft:
            return self.samples[0].isa_deviation_c
        if pressure_altitude_ft >= self.samples[-1].pressure_altitude_ft:
            return self.samples[-1].isa_deviation_c
        for low, high in zip(self.samples, self.samples[1:]):
            if pressure_altitude_ft <= high.pressure_altitude_ft:
                span = high.pressure_altitude_ft - low.pressure_altitude_ft
                frac = (pressure_altitude_ft - low.pressure_altitude_ft) / span
                return low.isa_deviation_c + frac * (
                    high.isa_deviation_c - low.isa_deviation_c
                )
        return self.samples[-1].isa_deviation_c  # unreachable; kept total

    def oat_at_pressure_altitude(self, pressure_altitude_ft: float) -> float:
        return isa_temperature_c(pressure_altitude_ft) + self.deviation_at(
            pressure_altitude_ft
        )

    def lapse_warnings(self) -> tuple[str, ...]:
        """Flag any pair of samples implying an impossible lapse rate.

        A warning, never a clamp. Inversions -- a marine layer, a cold valley
        floor under a warm morning -- are real and must stay plannable; it is
        only the steep side that says the numbers were entered wrong, and only
        well past the adiabatic rate at that.
        """
        notes: list[str] = []
        for low, high in zip(self.samples, self.samples[1:]):
            span = high.pressure_altitude_ft - low.pressure_altitude_ft
            if span <= 0:
                continue
            lapse = (low.oat_c - high.oat_c) / span * 1000.0
            if lapse > _IMPLAUSIBLE_LAPSE_C_PER_1000FT:
                notes.append(
                    f"temperatures at {low.pressure_altitude_ft:.0f} ft and "
                    f"{high.pressure_altitude_ft:.0f} ft pressure altitude imply "
                    f"{lapse:.1f} °C per 1000 ft, far steeper than the dry "
                    f"adiabatic rate of "
                    f"{DRY_ADIABATIC_C_PER_1000FT:.1f} — check the entries"
                )
        return tuple(notes)


# --- airspeed ------------------------------------------------------------

def tas_from_cas(cas_kt: float, density_altitude_ft: float) -> float:
    """The incompressible approximation, TAS = CAS / sqrt(sigma).
    In this case, we want to find out what the TAS would be, given a known
    CAS at a known density altitude
    """
    return cas_kt / math.sqrt(relative_density(density_altitude_ft))

def cas_from_tas(tas_kt: float, density_altitude_ft: float) -> float:
    """
    In this case, we want to find out what the CAS would read, given a 
    known TAS at a known density altitude
    """
    return tas_kt * math.sqrt(relative_density(density_altitude_ft))

