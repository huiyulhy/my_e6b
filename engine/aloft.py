"""Winds and temperatures aloft, from an Open-Meteo pressure-level forecast.

The surface tier (`engine/weather.py`) answers "what is the air doing at the
field". This is the other half: what it is doing at cruise, which is what
decides ground speed, fuel and every time on the log.

**Why a model at all.** The traditional answer is the FD winds-aloft forecast,
and this project deliberately does not use it: an FD level is one number for a
quarter of a state, issued for a handful of stations, and the wind on the coast
is not the wind over the valley twenty miles inland. A gridded model is
interpolated to the position asked about, which is the objection answered
rather than argued with. It is still a forecast, and it is labelled as one.

**Constant-pressure surfaces, not altitudes.** The model publishes on pressure
levels -- 850 hPa, 700 hPa -- and that turns out to be the convenient frame,
because a constant-pressure surface *has* a pressure altitude by definition.
The 850 hPa surface is at 4781 ft pressure altitude over the desert and over
the sea and in a hurricane; only its geometric height moves. So the temperature
profile is built with no altimeter setting involved at all, and cannot be wrong
by one.

The two products are therefore keyed differently, and it matters:

- **Temperature** is keyed by **pressure altitude**, exactly, from the level's
  own pressure. This is what the POH charts are read at.
- **Wind** is keyed by the level's **geopotential height** -- its true altitude
  MSL -- because a pilot holding 6500 ft indicated on a correct local setting
  is at about 6500 ft true, and that is the altitude the navlog looks the wind
  up at. Non-ISA air makes indicated and true diverge by a few hundred feet at
  light-aircraft altitudes; on a wind that is interpolated between levels
  thousands of feet apart, that is not a difference worth modelling.

**Levels below ground are dropped.** A model reports 1000 hPa everywhere,
including where the terrain is at 6000 ft, by extrapolating a fictional
atmosphere underneath the mountain. Truckee's 1000 hPa wind is not a wind. Any
level below the ground is discarded and said so in `notes`, rather than being
interpolated into the bottom of a climb. The ground is Open-Meteo's 90 m
elevation model rather than the weather model's own terrain -- see
`_terrain_elevation_ft` for why that is the stricter test.

Pure, like the rest of `engine/`: this parses and interpolates a payload
somebody else fetched. The network half is `server/wx_surface.fetch_aloft`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from engine.atmosphere import (
    TemperatureProfile,
    TemperatureSample,
    altitude_for_pressure_pa,
    isa_temperature_c,
    m_to_ft,
)
from engine.navlog import Wind, WindsAloft
from engine.weather import as_float, as_utc, nearest_hour_index, parse_iso_utc

__all__ = [
    "AloftForecast",
    "LevelSample",
    "PRESSURE_LEVELS_HPA",
    "hourly_fields",
    "levels_up_to",
    "parse_aloft",
]

# The pressure levels Open-Meteo publishes that a normally aspirated single
# ever flies in. 600 hPa is 13,800 ft, comfortably above this airplane's
# service ceiling; anything thinner is of no use to it and only makes the
# request longer.
PRESSURE_LEVELS_HPA: tuple[int, ...] = (1000, 975, 950, 925, 900, 850, 800, 700, 600)

# What is asked for at each level. Geopotential height is not optional: it is
# the only thing that puts the level at an altitude the navlog can look up.
_PER_LEVEL = (
    "temperature",
    "wind_speed",
    "wind_direction",
    "geopotential_height",
)

# A level whose height the model does not give cannot be placed, and a level
# below the terrain is a fiction. Both are dropped; see the module docstring.
_SURFACE_MARGIN_FT = 0.0


def levels_up_to(ceiling_ft: float) -> tuple[int, ...]:
    """The levels needed to bracket a flight up to `ceiling_ft`.

    One level above the ceiling as well as everything below it, so the top of
    the climb is interpolated between two levels rather than held flat from
    the last one under it.
    """
    kept: list[int] = []
    for hpa in PRESSURE_LEVELS_HPA:
        kept.append(hpa)
        if altitude_for_pressure_pa(hpa * 100.0) >= ceiling_ft:
            break
    return tuple(kept)


def hourly_fields(levels: tuple[int, ...]) -> tuple[str, ...]:
    """The `hourly=` names for a set of levels, in request order."""
    return tuple(f"{field}_{hpa}hPa" for hpa in levels for field in _PER_LEVEL)


@dataclass(frozen=True)
class LevelSample:
    """One pressure level over one point, at one hour."""

    pressure_hpa: float
    height_ft: float  # geopotential height, true altitude MSL
    wind: Wind | None = None
    oat_c: float | None = None

    @property
    def pressure_altitude_ft(self) -> float:
        """Exact, and independent of any altimeter setting.

        A constant-pressure surface is defined by its pressure, so its pressure
        altitude is a property of the level itself rather than of the day.
        """
        return altitude_for_pressure_pa(self.pressure_hpa * 100.0)

    @property
    def isa_deviation_c(self) -> float | None:
        if self.oat_c is None:
            return None
        return self.oat_c - isa_temperature_c(self.pressure_altitude_ft)


@dataclass(frozen=True)
class AloftForecast:
    """The column of air over one point, at one forecast hour.

    Levels are ordered bottom-up. `notes` carries what was discarded and why,
    on the same principle as the surface tier: a forecast that quietly threw
    something away is worse than one that says it did.
    """

    valid_time: datetime
    levels: tuple[LevelSample, ...] = ()
    terrain_elevation_ft: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def has_wind(self) -> bool:
        return any(level.wind is not None for level in self.levels)

    @property
    def has_temperature(self) -> bool:
        return any(level.oat_c is not None for level in self.levels)

    def winds(self) -> WindsAloft:
        """The wind profile, keyed by geopotential height.

        Empty where the forecast has no wind in it, which `WindsAloft` reads as
        calm -- the same thing a plan with nothing entered has always meant.
        """
        return WindsAloft(
            tuple(
                (level.height_ft, level.wind)
                for level in self.levels
                if level.wind is not None
            )
        )

    def temperature_samples(self) -> tuple[TemperatureSample, ...]:
        """The levels as observations, ready for `Conditions.temperatures_aloft`.

        **This, not `temperatures()`, is what a plan wants.** `build_navlog`
        builds one temperature curve for the whole route out of every sample it
        can find -- the field observations plus whatever is aloft -- so handing
        it a finished profile would see it thrown away and rebuilt. Handing it
        samples merges the two: the model column above, the METAR temperatures
        at the fields below, one curve through both.

        No conversion is involved, which is the payoff of keying by pressure
        altitude. A field temperature has to be converted through that
        station's altimeter setting before it can be compared with anything; a
        pressure level is already in the coordinate they all have in common.
        """
        return tuple(
            TemperatureSample(level.pressure_altitude_ft, level.isa_deviation_c)
            for level in self.levels
            if level.oat_c is not None
        )

    def temperatures(self, *, default_deviation_c: float = 0.0) -> TemperatureProfile:
        """The forecast column on its own, as a profile.

        For reading this forecast directly -- a temperature at an altitude,
        outside any plan. Inside a plan use `temperature_samples()`, which lets
        the fields' own observations into the same curve.

        `default_deviation_c` is only reached when the forecast carried no
        temperature at all, in which case this is the route-wide ISA deviation
        the pilot typed and the plan behaves exactly as it did before.
        """
        return TemperatureProfile.from_observations(
            self.temperature_samples(), default_deviation_c=default_deviation_c
        )

    def wind_at(self, altitude_ft: float) -> Wind:
        return self.winds().at(altitude_ft)

    def oat_at(self, pressure_altitude_ft: float) -> float | None:
        """Temperature at a pressure altitude, or None with nothing to read."""
        if not self.has_temperature:
            return None
        return self.temperatures().oat_at_pressure_altitude(pressure_altitude_ft)


def parse_aloft(
    payload: Any,
    target: datetime,
    *,
    levels: tuple[int, ...] = PRESSURE_LEVELS_HPA,
    index: int = 0,
) -> AloftForecast | None:
    """The pressure-level column nearest `target`, or None if there is none.

    `index` selects a location when the request batched several, matching
    `weather.parse_model_surface`: Open-Meteo returns a bare object for one
    coordinate and a list for many, and both are accepted.

    None -- rather than an empty forecast -- when the payload is unusable or
    holds no hour near the target. An empty forecast would read as "calm and
    standard", which is a claim about the day; None is the absence of one.
    """
    record = _record(payload, index)
    if record is None:
        return None
    hourly = record.get("hourly")
    if not isinstance(hourly, dict):
        return None
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        return None

    target = as_utc(target)
    slot = nearest_hour_index(times, target)
    if slot is None:
        return None

    def series(name: str) -> float | None:
        values = hourly.get(name)
        if not isinstance(values, list) or slot >= len(values):
            return None
        return as_float(values[slot])

    terrain_ft = _terrain_elevation_ft(record)
    notes: list[str] = []
    samples: list[LevelSample] = []
    underground: list[int] = []
    unplaced: list[int] = []

    for hpa in levels:
        height_m = series(f"geopotential_height_{hpa}hPa")
        if height_m is None:
            unplaced.append(hpa)
            continue
        height_ft = m_to_ft(height_m)
        # A level under the model's own terrain is an extrapolation into rock.
        if terrain_ft is not None and height_ft < terrain_ft - _SURFACE_MARGIN_FT:
            underground.append(hpa)
            continue
        samples.append(
            LevelSample(
                pressure_hpa=float(hpa),
                height_ft=height_ft,
                wind=_wind(
                    series(f"wind_direction_{hpa}hPa"), series(f"wind_speed_{hpa}hPa")
                ),
                oat_c=series(f"temperature_{hpa}hPa"),
            )
        )

    if underground:
        notes.append(
            f"{_levels_phrase(underground)} lie below the model's ground at "
            f"{terrain_ft:.0f} ft and were dropped"
        )
    if unplaced:
        notes.append(
            f"{_levels_phrase(unplaced)} came with no height and could not be placed"
        )
    if not samples:
        notes.append("no usable level in this forecast")

    samples.sort(key=lambda level: level.height_ft)
    return AloftForecast(
        valid_time=parse_iso_utc(times[slot]) or target,
        levels=tuple(samples),
        terrain_elevation_ft=terrain_ft,
        notes=tuple(notes),
    )


def _record(payload: Any, index: int) -> dict | None:
    if isinstance(payload, list):
        if index >= len(payload):
            return None
        payload = payload[index]
    return payload if isinstance(payload, dict) else None


def _terrain_elevation_ft(record: dict) -> float | None:
    """The ground height at this point, in feet.

    Open-Meteo reports it in metres beside the forecast, off a 90 m digital
    elevation model -- the real terrain, not the weather model's smoothed
    version of it, and the same value whichever model is asked. At Truckee it
    comes back as 5899 ft against a field elevation of 5900 ft.

    That is the better number for this. A global model flattens the Sierra into
    something a couple of thousand feet lower, so levels it considers to be
    above its own ground can still be underneath the actual mountain. The
    question here is whether a level is below ground the airplane has to clear,
    and the DEM answers that one.
    """
    metres = as_float(record.get("elevation"))
    return None if metres is None else m_to_ft(metres)


def _wind(from_deg: float | None, speed_kt: float | None) -> Wind | None:
    """A wind, or None when either half is missing.

    Half a wind is not one: a direction with no speed would read as calm from
    the north, and a speed with no direction has nowhere to blow.
    """
    if from_deg is None or speed_kt is None:
        return None
    return Wind(from_deg % 360.0, speed_kt)


def _levels_phrase(levels: list[int]) -> str:
    named = ", ".join(f"{hpa} hPa" for hpa in levels)
    return f"the {named} {'levels' if len(levels) > 1 else 'level'}"
