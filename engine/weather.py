"""Decoded weather for a point on the ground
Fetching takes place in server/wx_surface.py

3 possible sources:
1. METAR - real time observation (wind, vis, cloud, temperature, pressure)
2. TAF - 24-30 hour forecast (wind and vis only)
3. Model - GFS global or HRRR (high res) via open meteo. Weather
for a given point (pos and alt) at any time

resolve_surface blends these observations and selects a value
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from engine.navlog import Wind

__all__ = [
    "HPA_PER_INHG",
    "OBSCURATION_COVERS",
    "Ceiling",
    "Source",
    "SurfaceWeather",
    "WeatherUnavailable",
    "hpa_to_inhg",
    "parse_metar",
    "parse_metar_history",
    "parse_model_surface",
    "parse_taf",
    "resolve_surface",
]

class WeatherUnavailable(RuntimeError):
    """No source could be reached, or none of them answered for this station.

    Raised by the fetchers rather than in here, but defined here so callers
    import the weather vocabulary from one place.
    """


# The Aviation Weather Center reports `altim` in hectopascals -- 1012.6, not
# 29.92 -- while the raw METAR text beside it says `A2990` and every altimeter
# in a US cockpit is set in inches.
HPA_PER_INHG = 33.863886666667

# Acceptable limits for altimeter setting
_MIN_PLAUSIBLE_INHG = 25.0
_MAX_PLAUSIBLE_INHG = 32.5


def hpa_to_inhg(hpa: float) -> float:
    inhg = hpa / HPA_PER_INHG
    if not _MIN_PLAUSIBLE_INHG <= inhg <= _MAX_PLAUSIBLE_INHG:
        raise ValueError(
            f"altimeter setting {inhg:.2f} inHg (from {hpa:g} hPa) is outside "
            f"{_MIN_PLAUSIBLE_INHG}-{_MAX_PLAUSIBLE_INHG} inHg; check the units"
        )
    return inhg


class Source(StrEnum):
    """Where one field's value came from.

    Carried per field rather than per report, because a resolved observation
    routinely mixes two sources and the mix is the interesting part.
    """

    METAR = "metar"
    TAF = "taf"
    MODEL = "model"
    NEAREST_TAF = "nearest_taf"


@dataclass(frozen=True)
class SurfaceWeather:
    """The air at one field at a given time. Each measurement is optional
    (e.g. TAF contains missing pressure and temp)
    """

    station: str
    valid_time: datetime  # the hour this describes, always UTC
    wind: Wind | None = None
    gust_kt: float | None = None
    visibility_sm: float | None = None
    ceiling_ft_agl: float | None = None
    # What made that height a ceiling: BKN, OVC, or the OVX/VV of an
    # obscuration. Carried because the go/no-go treats them differently -- an
    # overcast has to sit high enough to fly a pattern under, and a vertical
    # visibility means there is no sky to fly under at all.
    ceiling_cover: str | None = None
    # Whether the report carried a sky-condition group at all. A clear sky and
    # a source that does not observe cloud both leave `ceiling_ft_agl` empty,
    # and they are opposite answers: the first is VFR, the second is unknown.
    sky_reported: bool = False
    oat_c: float | None = None
    altimeter_inhg: float | None = None
    sources: Mapping[str, Source] = None  # type: ignore[assignment]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sources is None:
            object.__setattr__(self, "sources", {})

    @property
    def has_field_conditions(self) -> bool:
        """True when a density altitude can be computed from this.
        The go/no-go check needs both halves
        """
        return self.oat_c is not None and self.altimeter_inhg is not None


# --- payload helpers ------------------------------------------------------
# The AWC and Open-Meteo payloads are loosely typed in ways that matter:
# `visib` arrives as `10` on one observation and `"10+"` on the next, and any
# numeric field may be absent or null. These normalise without inventing.


def as_float(value: Any) -> float | None:
    """A float, or None -- never a guess."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _visibility_sm(value: Any) -> float | None:
    """Statute miles from AWC's `visib`, which is a number or a string.

    `"10+"` and `"6+"` mean "at least this much", which for planning is the
    value itself: the distances and minima it feeds are all lower bounds.
    Fractions like `"1 1/2"` appear in low visibility and are parsed rather
    than dropped, since those are exactly the reports a go/no-go turns on.
    """
    number = as_float(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip("+").strip()
    if not text:
        return None
    total = 0.0
    for part in text.split():
        if "/" in part:
            numerator, _, denominator = part.partition("/")
            try:
                total += float(numerator) / float(denominator)
            except (ValueError, ZeroDivisionError):
                return None
        else:
            try:
                total += float(part)
            except ValueError:
                return None
    return total


# A ceiling is the lowest broken or overcast layer -- scattered and few are not
# ceilings. Vertical visibility in an obscuration counts as one.
_CEILING_COVERS = frozenset({"BKN", "OVC", "OVX", "VV"})


# The covers that mean the sky is not visible at all, only a vertical
# visibility into it. AWC writes `VV` in a TAF and `OVX` in a METAR.
OBSCURATION_COVERS = frozenset({"OVX", "VV"})


@dataclass(frozen=True)
class Ceiling:
    """The lowest ceiling in a report, and what kind of ceiling it is."""

    ft_agl: float | None = None
    cover: str | None = None
    # True when the report said something about the sky, including "clear".
    reported: bool = False


def _ceiling(clouds: Any, vert_vis: Any = None) -> Ceiling:
    """The lowest broken, overcast or obscured layer, with its cover.

    A vertical visibility is reported as a height into an obscuration rather
    than as a cloud base, and it is the ceiling whenever it is the lowest
    thing in the report -- you cannot fly VFR under a sky you cannot see.
    """
    vertical = as_float(vert_vis)
    reported = isinstance(clouds, list) or vertical is not None
    lowest: float | None = None
    cover: str | None = None
    if isinstance(clouds, list):
        for layer in clouds:
            if not isinstance(layer, dict):
                continue
            this_cover = str(layer.get("cover") or "").upper()
            if this_cover not in _CEILING_COVERS:
                continue
            base = as_float(layer.get("base"))
            # An obscuration group with no height on it is still an
            # obscuration: it is reported, it is a ceiling, and only its
            # height is missing.
            if base is None:
                if this_cover in OBSCURATION_COVERS and cover is None:
                    cover = this_cover
                continue
            if lowest is None or base < lowest:
                lowest, cover = base, this_cover
    if vertical is not None and (lowest is None or vertical < lowest):
        lowest, cover = vertical, "VV"
    return Ceiling(ft_agl=lowest, cover=cover, reported=reported)


def _ceiling_ft_agl(clouds: Any, vert_vis: Any = None) -> float | None:
    """Just the height, for callers that do not care what kind it is."""
    return _ceiling(clouds, vert_vis).ft_agl


def _wind(wdir: Any, wspd: Any) -> Wind | None:
    """A wind, or None when the report does not give one.

    AWC uses `wdir` 0 with a speed for calm and the string `"VRB"` for a
    variable direction. A variable direction is a real report but not a usable
    one for a wind triangle, so it becomes `None` rather than a made-up
    heading; the speed alone cannot be flown.
    """
    speed = as_float(wspd)
    if speed is None:
        return None
    direction = as_float(wdir)
    if direction is None:
        # "VRB" or absent. Calm is the one case where no direction is fine.
        return Wind(0.0, 0.0) if speed == 0.0 else None
    return Wind(direction % 360.0, speed)


def _utc(epoch: Any) -> datetime | None:
    seconds = as_float(epoch)
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, tz=UTC)


def _first(payload: Any) -> dict | None:
    """AWC returns a list even for a single station; Open-Meteo returns a dict."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                return entry
    return None


# --- METAR ----------------------------------------------------------------


def parse_metar(payload: Any) -> SurfaceWeather | None:
    """One observation from the AWC METAR endpoint.

    Takes either the decoded list the API returns or a single record from it.
    `None` when the payload holds no usable observation, which is what an
    empty list from a station that is not reporting looks like.
    """
    record = _first(payload)
    if record is None:
        return None
    observed = _utc(record.get("obsTime"))
    if observed is None:
        return None

    altimeter: float | None = None
    hpa = as_float(record.get("altim"))
    if hpa is not None:
        try:
            altimeter = hpa_to_inhg(hpa)
        except ValueError:
            altimeter = None

    wind = _wind(record.get("wdir"), record.get("wspd"))
    oat = as_float(record.get("temp"))
    visibility = _visibility_sm(record.get("visib"))
    ceiling = _ceiling(record.get("clouds"))

    sources = {
        name: Source.METAR
        for name, value in (
            ("wind", wind),
            ("gust_kt", as_float(record.get("wgst"))),
            ("visibility_sm", visibility),
            ("ceiling_ft_agl", ceiling.ft_agl),
            ("oat_c", oat),
            ("altimeter_inhg", altimeter),
        )
        if value is not None
    }

    return SurfaceWeather(
        station=str(record.get("icaoId") or "").upper(),
        valid_time=observed,
        wind=wind,
        gust_kt=as_float(record.get("wgst")),
        visibility_sm=visibility,
        ceiling_ft_agl=ceiling.ft_agl,
        ceiling_cover=ceiling.cover,
        sky_reported=ceiling.reported,
        oat_c=oat,
        altimeter_inhg=altimeter,
        sources=sources,
    )


def parse_metar_history(payload: Any) -> dict[str, list[SurfaceWeather]]:
    """Many observations, many stations, grouped by station and sorted by time.

    The AWC METAR endpoint takes `hours=` and a comma-separated station list
    and returns every observation in one flat list. This is what
    `tools/plot_altimeter_trend.py` reads, and it is here rather than in the
    tool because it is the same parse as a single observation and a trend plot
    that disagreed with the planner about a unit would be worse than no plot.
    """
    if not isinstance(payload, list):
        return {}
    grouped: dict[str, list[SurfaceWeather]] = {}
    for record in payload:
        observation = parse_metar(record)
        if observation is None or not observation.station:
            continue
        grouped.setdefault(observation.station, []).append(observation)
    for observations in grouped.values():
        observations.sort(key=lambda item: item.valid_time)
    return grouped


# --- TAF ------------------------------------------------------------------

# A TEMPO or PROB group describes a condition expected to come and go within
# the period. It is real and worth reading, but it is not the planning value:
# taking a TEMPO wind as *the* wind would plan every flight for its worst
# half-hour. They are reported in `notes` and not applied.
_TRANSIENT_CHANGES = frozenset({"TEMPO", "PROB", "PROB30", "PROB40"})


def parse_taf(payload: Any, target: datetime) -> SurfaceWeather | None:
    """The forecast block covering `target`, if the TAF has one.

    Blocks are matched on `timeFrom <= target < timeTo`. The **last** matching
    non-transient block wins: `FM` and `BECMG` groups are amendments to what
    came before, and they are published in order, so the latest one that has
    started is the one in force.

    Returns `None` when the TAF does not cover `target` at all -- a target
    beyond the end of the valid period, or before it was issued. That is a
    real answer, not a failure: it means this source cannot speak to that time
    and the caller should fall back rather than stretch the last block.

    Note what is *not* in the result: `oat_c` and `altimeter_inhg` are always
    `None` here, because a TAF does not carry them.
    """
    record = _first(payload)
    if record is None:
        return None
    forecasts = record.get("fcsts")
    if not isinstance(forecasts, list):
        return None

    target = as_utc(target)
    station = str(record.get("icaoId") or "").upper()

    chosen: dict | None = None
    notes: list[str] = []
    for block in forecasts:
        if not isinstance(block, dict):
            continue
        start = _utc(block.get("timeFrom"))
        end = _utc(block.get("timeTo"))
        if start is None or end is None or not start <= target < end:
            continue
        change = str(block.get("fcstChange") or "").upper()
        if change in _TRANSIENT_CHANGES:
            probability = block.get("probability")
            notes.append(
                f"{change}{'' if probability is None else f' {probability}%'} "
                f"group in force at this time; not applied"
            )
            continue
        chosen = block

    if chosen is None:
        return None

    wind = _wind(chosen.get("wdir"), chosen.get("wspd"))
    visibility = _visibility_sm(chosen.get("visib"))
    ceiling = _ceiling(chosen.get("clouds"), chosen.get("vertVis"))
    gust = as_float(chosen.get("wgst"))

    sources = {
        name: Source.TAF
        for name, value in (
            ("wind", wind),
            ("gust_kt", gust),
            ("visibility_sm", visibility),
            ("ceiling_ft_agl", ceiling.ft_agl),
        )
        if value is not None
    }

    return SurfaceWeather(
        station=station,
        valid_time=target,
        wind=wind,
        gust_kt=gust,
        visibility_sm=visibility,
        ceiling_ft_agl=ceiling.ft_agl,
        ceiling_cover=ceiling.cover,
        sky_reported=ceiling.reported,
        oat_c=None,  # a TAF does not carry temperature
        altimeter_inhg=None,  # nor an altimeter setting
        sources=sources,
        notes=tuple(notes),
    )


# --- model ----------------------------------------------------------------


def parse_model_surface(
    payload: Any, target: datetime, *, station: str = "", index: int = 0
) -> SurfaceWeather | None:
    """The Open-Meteo hourly surface row nearest `target`.

    `index` selects a location when the request batched several; Open-Meteo
    returns a bare object for one coordinate and a list for many, and this
    accepts either.

    The altimeter setting comes from `pressure_msl`, **not** `surface_pressure`.
    An altimeter setting is a sea-level-reduced pressure by definition; using
    station pressure would subtract the field elevation a second time when
    `pressure_altitude` applies it, reading a high field as far higher still.
    """
    if isinstance(payload, list):
        if index >= len(payload):
            return None
        record = payload[index]
    else:
        record = payload
    if not isinstance(record, dict):
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

    altimeter: float | None = None
    hpa = series("pressure_msl")
    if hpa is not None:
        try:
            altimeter = hpa_to_inhg(hpa)
        except ValueError:
            altimeter = None

    wind = _wind(series("wind_direction_10m"), series("wind_speed_10m"))
    oat = series("temperature_2m")
    gust = series("wind_gusts_10m")

    sources = {
        name: Source.MODEL
        for name, value in (
            ("wind", wind),
            ("gust_kt", gust),
            ("oat_c", oat),
            ("altimeter_inhg", altimeter),
        )
        if value is not None
    }

    valid = parse_iso_utc(times[slot]) or target
    return SurfaceWeather(
        station=station.upper(),
        valid_time=valid,
        wind=wind,
        gust_kt=gust,
        oat_c=oat,
        altimeter_inhg=altimeter,
        sources=sources,
    )


def parse_iso_utc(text: Any) -> datetime | None:
    """Open-Meteo hours are naive ISO strings; the request pins them to GMT.

    Public, with `as_float`, `as_utc` and `nearest_hour_index`: `engine/aloft.py`
    reads the same hourly payload a level at a time, and a second copy of "an
    hour more than an hour away is not this hour" would be a correctness fork
    rather than a convenience.
    """
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def nearest_hour_index(times: list, target: datetime) -> int | None:
    """The hourly slot closest to `target`, refusing a distant one.

    Model output is hourly, so the worst honest error is half an hour. A
    target outside the returned window entirely would otherwise silently snap
    to the first or last hour available, which is how a forecast for tomorrow
    afternoon quietly becomes one for this morning.
    """
    best: int | None = None
    best_gap: float | None = None
    for slot, text in enumerate(times):
        moment = parse_iso_utc(text)
        if moment is None:
            continue
        gap = abs((moment - target).total_seconds())
        if best_gap is None or gap < best_gap:
            best, best_gap = slot, gap
    if best is None or best_gap is None or best_gap > 3600.0:
        return None
    return best


# --- the decider ----------------------------------------------------------


def as_utc(moment: datetime) -> datetime:
    """A naive time is read as UTC -- every source here publishes in Zulu."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


# How far ahead an observation still counts as "now". A METAR is issued hourly
# and a special can land between, so anything within the hour is current; past
# that the observation is stale and the forecast is the better answer even for
# the current time.
_METAR_STILL_CURRENT_MIN = 75.0


def resolve_surface(
    *,
    station: str,
    target: datetime | None = None,
    now: datetime | None = None,
    metar: SurfaceWeather | None = None,
    taf: SurfaceWeather | None = None,
    model: SurfaceWeather | None = None,
    taf_station: str | None = None,
    taf_distance_nm: float | None = None,
) -> SurfaceWeather:
    """Combine the three sources into one report, field by field.

    The rule, in order:

    1. **Target is now (or omitted), and the METAR is current** -- the METAR
       wins on every field it publishes. An observation beats a forecast of
       the same moment, always.
    2. **Target is in the future** -- the TAF supplies wind, gust, visibility
       and ceiling, and the **model supplies temperature and altimeter
       setting**, which the TAF structurally lacks. This is the case that
       makes a target-time go/no-go possible at all.
    3. **No TAF covers the target** -- the model supplies everything it can,
       tagged as a model rather than an observation.
    4. **Anything still missing stays `None`.** Nothing is carried forward
       from a stale observation or interpolated across a gap.

    Two things are not resolved field by field. **The gust belongs to the
    wind**, and **the sky is one observation**: a ceiling height from the TAF
    wearing a cover code from the METAR would describe a sky that neither
    source reported, so the height, the cover and the fact that cloud was
    observed at all move together or not at all.

    On the gust:
    A METAR that gives a wind and no gust group is stating that there is no
    gust, not leaving a hole for a model to fill -- KHAF reporting `00000KT`
    while the model says 9 kt would otherwise resolve to "calm, gusting 9",
    a peak that appeared in neither source and that the go/no-go would then
    resolve onto a runway as crosswind. So a gust from a weaker source than
    the wind it would be attached to is dropped, and the drop is noted.

    `taf_station` and `taf_distance_nm` describe a TAF borrowed from a
    neighbouring field. Borrowing is legitimate -- only around 600 US airports
    publish a TAF, and most of the fields a light aircraft uses do not -- but
    it is recorded in `notes` and marked `NEAREST_TAF` rather than passed off
    as this field's own forecast.
    """
    station = station.upper()
    now = as_utc(now) if now is not None else datetime.now(tz=UTC)
    target = as_utc(target) if target is not None else now

    borrowed = bool(taf_station) and taf_station.upper() != station
    taf_source = Source.NEAREST_TAF if borrowed else Source.TAF

    notes: list[str] = []
    ahead_min = (target - now).total_seconds() / 60.0
    metar_is_current = (
        metar is not None and abs((target - metar.valid_time).total_seconds()) / 60.0
        <= _METAR_STILL_CURRENT_MIN
    )
    # Ordered weakest first: later sources overwrite earlier ones per field.
    if ahead_min <= _METAR_STILL_CURRENT_MIN and metar_is_current:
        ladder = ((model, Source.MODEL), (taf, taf_source), (metar, Source.METAR))
    else:
        # The METAR is left out entirely rather than used as a last resort.
        # It is an observation of a different time, and the fields it would be
        # filling here are exactly the ones that move: an afternoon takeoff
        # computed on the morning's temperature reads short on the hottest
        # day of the year. Better to return None and say why.
        ladder = ((model, Source.MODEL), (taf, taf_source))
        if metar is not None:
            age_min = (target - metar.valid_time).total_seconds() / 60.0
            if age_min > _METAR_STILL_CURRENT_MIN:
                notes.append(
                    f"latest {station} observation is {age_min / 60.0:.1f} h "
                    f"before this time and was not used"
                )

    fields = ("wind", "gust_kt", "visibility_sm", "oat_c", "altimeter_inhg")
    values: dict[str, Any] = dict.fromkeys(fields)
    sources: dict[str, Source] = {}
    # The sky group, taken whole from the strongest source that observed one.
    sky: dict[str, Any] = {
        "ceiling_ft_agl": None,
        "ceiling_cover": None,
        "sky_reported": False,
    }

    for report, source in ladder:
        if report is None:
            continue
        notes.extend(report.notes)
        for name in fields:
            value = getattr(report, name)
            if value is None:
                continue
            values[name] = value
            sources[name] = source
        if report.sky_reported:
            sky = {
                "ceiling_ft_agl": report.ceiling_ft_agl,
                "ceiling_cover": report.ceiling_cover,
                "sky_reported": True,
            }
            # Only a ceiling has a source to name. A reported clear sky is an
            # answer, but it is not a value any box is filled from.
            sources.pop("ceiling_ft_agl", None)
            if report.ceiling_ft_agl is not None:
                sources["ceiling_ft_agl"] = source
    values.update(sky)

    # The gust travels with the wind that was measured alongside it, or not
    # at all. Same report, or nothing: two sources of one wind is not a wind.
    if values["gust_kt"] is not None and sources.get("gust_kt") != sources.get("wind"):
        dropped, wind_source = sources.pop("gust_kt"), sources.get("wind")
        notes.append(
            f"a {dropped} gust of {values['gust_kt']:.0f} kt was dropped: "
            + (
                f"the wind it would attach to came from the {wind_source} instead"
                if wind_source is not None
                else "there is no wind for this time to attach it to"
            )
        )
        values["gust_kt"] = None

    if borrowed and any(source is Source.NEAREST_TAF for source in sources.values()):
        distance = "" if taf_distance_nm is None else f", {taf_distance_nm:.0f} nm away"
        notes.append(
            f"{station} publishes no TAF; forecast taken from "
            f"{taf_station.upper()}{distance}"
        )

    if values["oat_c"] is None or values["altimeter_inhg"] is None:
        notes.append(
            "no temperature and altimeter setting for this time; "
            "density altitude and runway distances cannot be computed"
        )

    return SurfaceWeather(
        station=station,
        valid_time=target,
        sources=sources,
        notes=tuple(dict.fromkeys(notes)),
        **values,
    )
