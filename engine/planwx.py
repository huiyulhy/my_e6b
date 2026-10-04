"""The weather list beside the navigation log, and the loop that settles them.

This part manages the weather list by iterating in a forward and backward pass
1 list: the one drawn by the pilot
2nd list: the weather list that the plane flies through, so we can get performance
Two lists, one entry per leg the pilot drew: the navlog rows, and the weather
each of them was planned in. `solve` produces both, and they agree -- which is
the whole difficulty, because each is an input to the other.

**Why the two ends, and why the worse of them.** A leg is not a point. The
forecast over the airport it starts at and the forecast over the airport it
ends at are two different columns of air, and the aeroplane flies through
both. Planning on either alone is a guess about which half of the leg matters;
planning on the average is a wind that was forecast nowhere. So each leg is
costed under both and planned in whichever costs more fuel. A plan that comes
in early on the day is a good day; a plan that comes in late is a diversion.


1. Plan with no forecast, to learn roughly when each waypoint is reached.
2. Choose each leg's weather from the forecasts at its two ends, at those
   times, taking whichever costs more.
3. Plan again on that weather, which moves the times -- and the tops of climb.
4. Re-choose. If nothing changed, the two lists agree and it is done.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta

from engine import navlog as nl
from engine.atmosphere import TemperatureSample
from engine.geo import LatLon, WindTooStrong, solve_wind_triangle
from engine.preflight import Margins

__all__ = [
    "MAX_PASSES",
    "ForecastHour",
    "LegWeather",
    "PointForecast",
    "SolvedPlan",
    "solve",
]

# Max allowable number of forward and backward passes
MAX_PASSES = 6
_UNFLYABLE = math.inf


@dataclass(frozen=True)
class ForecastHour:
    """One hour of forecast over one point: the column, wind and temperature.

    The two travel together because they were reported together. Which hour
    this is matters as much as what is in it -- see `PointForecast.at`.
    """

    valid_time: datetime
    winds: nl.WindsAloft = dataclass_field(default_factory=nl.WindsAloft.calm)
    temperatures: tuple[TemperatureSample, ...] = ()


@dataclass(frozen=True)
class PointForecast:
    """Every hour that was fetched over one point of the route.

    A whole series rather than the single hour the plan will use, because
    which hour it uses is not known until the plan is built, and the plan is
    not built until the weather is chosen. Fetching the window once and
    re-reading it is what lets the loop turn without going back to the
    network.
    """

    name: str
    position: LatLon
    hours: tuple[ForecastHour, ...] = ()

    def at(self, when: datetime | None) -> ForecastHour | None:
        """The forecast hour nearest a time, or `None` if nothing was fetched.

        With no time -- a plan with no off-blocks on it -- the earliest hour
        is used, which is the one the fetch was anchored to: now.
        """
        if not self.hours:
            return None
        if when is None:
            return min(self.hours, key=lambda hour: hour.valid_time)
        return min(
            self.hours,
            key=lambda hour: abs((hour.valid_time - _utc(when)).total_seconds()),
        )


@dataclass(frozen=True)
class LegWeather:
    """The weather one drawn leg was planned in, and why that one.

    The second of the two lists. One entry per leg the pilot drew, in route
    order, including legs no forecast was available for -- a gap in the list
    would silently renumber it against the first.
    """

    leg: int  # index into the legs the pilot drew
    from_name: str
    to_name: str

    # "start" | "end" | "" when there was no forecast to choose from.
    chosen: str = ""
    valid_time: datetime | None = None
    winds: nl.WindsAloft = dataclass_field(default_factory=nl.WindsAloft.calm)
    temperatures: tuple[TemperatureSample, ...] = ()

    # What each end was worth over this leg, in gallons. `None` where that end
    # had no forecast, or where there was no plan yet to cost it against.
    start_fuel_gal: float | None = None
    end_fuel_gal: float | None = None
    # An end whose wind no heading can hold the course in. Not a cost -- there
    # is no number for it -- and not something to plan on either, so it is
    # recorded rather than chosen. See `_choose_one`.
    start_unflyable: bool = False
    end_unflyable: bool = False
    # The wind each end offered at the leg's own altitude, for the panel.
    start_wind: nl.Wind | None = None
    end_wind: nl.Wind | None = None
    note: str = ""

    @property
    def has_forecast(self) -> bool:
        return bool(self.chosen)

    @property
    def wind(self) -> nl.Wind | None:
        """The wind that was chosen, at the altitude the leg is flown at."""
        return self.start_wind if self.chosen == "start" else self.end_wind

    @property
    def key(self) -> tuple:
        """What has to stop changing before the two lists have settled.

        The choice, not the numbers. Which end won and which hour it was read
        at are both discrete, so this either changes on a pass or it does not.
        """
        return (self.chosen, self.valid_time)


@dataclass(frozen=True)
class SolvedPlan:
    """The two lists, and how they were arrived at."""

    navlog: nl.Navlog
    weather: tuple[LegWeather, ...]
    passes: int
    settled: bool

    @property
    def notes(self) -> tuple[str, ...]:
        """What a pilot should know about how this was reached."""
        if not self.weather:
            return ()
        if not self.settled:
            return (
                (
                    f"the weather and the navigation log did not settle in "
                    f"{self.passes} passes: the chosen forecast keeps changing "
                    f"as the times move. The plan shown is the last pass -- "
                    f"check the wind on each row against the forecast you were "
                    f"briefed on"
                ),
            )
        return ()


def solve(
    waypoints: list[nl.Waypoint],
    cruise_altitude_ft: float,
    aircraft: nl.Aircraft | None = None,
    conditions: nl.Conditions | None = None,
    *,
    forecasts: tuple[PointForecast, ...] = (),
    off_blocks: datetime | None = None,
    overrides: dict[int, nl.LegOverride] | None = None,
    margins: Margins | None = None,
    planning_mode: str = "manual",
    max_passes: int = MAX_PASSES,
    segment_overrides: nl.SegmentOverrides | None = None,
) -> SolvedPlan:
    """Build a navlog and the weather list it agrees with.

    `forecasts` is one series per waypoint the pilot drew, in route order --
    the same order and length as the route with its generated points removed.
    A short list, or one with empty series in it, is not an error: the legs it
    cannot cover are planned on whatever was typed on their rows, and on calm
    below that, which is what a plan with no weather on it has always meant.

    With no forecasts at all this is `build_navlog` with one extra comparison
    that finds nothing, and it returns on the first pass.
    """
    conditions = conditions or nl.Conditions()
    drawn = nl.strip_generated(waypoints)

    def plan(weather: tuple[LegWeather, ...]) -> nl.Navlog:
        return nl.build_navlog(
            waypoints,
            cruise_altitude_ft,
            aircraft,
            _with_weather(conditions, drawn, weather),
            overrides=overrides,
            margins=margins,
            planning_mode=planning_mode,
            segment_overrides=segment_overrides,
        )

    # Pass zero: no forecast at all, purely to learn roughly when each
    # waypoint is reached. Its own winds are whatever the pilot typed.
    weather: tuple[LegWeather, ...] = ()
    navlog = plan(weather)
    if not forecasts or not any(series.hours for series in forecasts):
        return SolvedPlan(navlog=navlog, weather=(), passes=1, settled=True)

    for attempt in range(1, max_passes + 1):
        chosen = _choose(drawn, navlog, forecasts, off_blocks)
        if _same(chosen, weather):
            # The choice survived the plan it produced. The two lists agree.
            return SolvedPlan(
                navlog=navlog, weather=chosen, passes=attempt, settled=True
            )
        weather = chosen
        navlog = plan(weather)

    # Out of passes. The last plan is returned rather than nothing -- it is a
    # real plan on a real forecast, just not one that proved stable -- and
    # `SolvedPlan.notes` says so.
    return SolvedPlan(
        navlog=navlog, weather=weather, passes=max_passes, settled=False
    )


# --- choosing -------------------------------------------------------------


def _choose(
    drawn: list[nl.Waypoint],
    navlog: nl.Navlog,
    forecasts: tuple[PointForecast, ...],
    off_blocks: datetime | None,
) -> tuple[LegWeather, ...]:
    """The weather list for one pass, read off the navlog of the pass before.

    The navlog supplies two things: when each waypoint is reached, which picks
    the forecast hour at each end, and what each leg costs, which is how the
    two ends are compared.
    """
    times = _arrival_times(navlog, drawn, off_blocks)
    rows = nl.rows_by_drawn_leg(navlog.legs, drawn)

    chosen: list[LegWeather] = []
    for index in range(len(drawn) - 1):
        start = _series(forecasts, index)
        end = _series(forecasts, index + 1)
        chosen.append(
            _choose_one(
                index=index,
                from_name=drawn[index].name,
                to_name=drawn[index + 1].name,
                rows=rows.get(index, []),
                start=None if start is None else start.at(times[index]),
                end=None if end is None else end.at(times[index + 1]),
            )
        )
    return tuple(chosen)


def _choose_one(
    *,
    index: int,
    from_name: str,
    to_name: str,
    rows: list[nl.Leg],
    start: ForecastHour | None,
    end: ForecastHour | None,
) -> LegWeather:
    """One leg, costed at both ends, planned in the more expensive.

    Fuel rather than time, because fuel is what a reserve is measured in and
    the two can disagree: a leg flown higher is slower over the ground and
    cheaper per hour. Where the two ends cost the same -- which they do
    whenever the forecast is uniform, and on the first pass before there is
    anything to cost against -- the leg is planned on the forecast for the end
    it is *reached at*, since that is the later and less certain of the two.

    An end whose wind no heading can hold the course in is not chosen, even
    though it is the most expensive thing there is. Planning on it would make
    the route unbuildable and leave the pilot looking at an error instead of
    at a plan -- so the other end is used and the fact is written on the leg,
    which says strictly more than the refusal would have.
    """
    identity = {"leg": index, "from_name": from_name, "to_name": to_name}

    if start is None and end is None:
        return LegWeather(**identity, note="no forecast over this leg")

    start_fuel = None if start is None else _fuel_gal(rows, start.winds)
    end_fuel = None if end is None else _fuel_gal(rows, end.winds)
    start_out = start_fuel == _UNFLYABLE
    end_out = end_fuel == _UNFLYABLE
    measured = {
        "start_fuel_gal": None if start_out else start_fuel,
        "end_fuel_gal": None if end_out else end_fuel,
        "start_unflyable": start_out,
        "end_unflyable": end_out,
        "start_wind": None if start is None else _wind_over(rows, start.winds),
        "end_wind": None if end is None else _wind_over(rows, end.winds),
    }
    cannot = _unflyable_note(from_name, to_name, start_out, end_out)

    if start_out and end_out:
        return LegWeather(**identity, **measured, note=cannot)
    if start is None or start_out:
        pick, why = "end", f"only {to_name} has a forecast to plan on"
    elif end is None or end_out:
        pick, why = "start", f"only {from_name} has a forecast to plan on"
    elif start_fuel > end_fuel:
        pick = "start"
        why = (
            f"{from_name}'s forecast costs {start_fuel:.1f} gal against "
            f"{to_name}'s {end_fuel:.1f}"
        )
    elif end_fuel > start_fuel:
        pick = "end"
        why = (
            f"{to_name}'s forecast costs {end_fuel:.1f} gal against "
            f"{from_name}'s {start_fuel:.1f}"
        )
    else:
        pick = "end"
        why = (
            f"both ends cost the same; planned on {to_name}'s forecast, the "
            f"later of the two"
        )

    hour = start if pick == "start" else end
    return LegWeather(
        **identity,
        **measured,
        chosen=pick,
        valid_time=hour.valid_time,
        winds=hour.winds,
        temperatures=hour.temperatures,
        note="; ".join(part for part in (cannot, why) if part),
    )


def _unflyable_note(
    from_name: str, to_name: str, start_out: bool, end_out: bool
) -> str:
    """Said plainly, because it is the most important thing on the leg."""
    out = [name for name, gone in ((from_name, start_out), (to_name, end_out)) if gone]
    if not out:
        return ""
    which = " and ".join(out)
    tail = (
        "neither end can be planned on, so this leg falls back to the wind "
        "typed on its rows"
        if start_out and end_out
        else "planned on the other end instead"
    )
    return (
        f"the forecast at {which} cannot be flown at all -- the wind exceeds "
        f"what the aeroplane can hold the course against; {tail}"
    )


def _fuel_gal(rows: list[nl.Leg], winds: nl.WindsAloft) -> float | None:
    """What this leg's rows would burn in a given wind.

    Read off the rows as they already stand: their distance, their altitude,
    their true airspeed and the fuel flow the POH gave them. Only the ground
    speed is recomputed, because only the wind is being varied -- the power
    setting is the same under either forecast, so the chart does not need
    reading again to compare them.

    `None` with no rows to cost, which is every leg on the first pass.
    """
    if not rows:
        return None
    total = 0.0
    for row in rows:
        wind = winds.at(row.altitude_ft)
        minutes = _minutes(row, wind)
        if minutes == _UNFLYABLE:
            return _UNFLYABLE
        # The row's own fuel flow, recovered from what it was charged. A row
        # of no length has none to recover and costs nothing either way.
        if row.ete_min <= 0.0:
            continue
        total += row.fuel_gal * minutes / row.ete_min
    return total


def _minutes(row: nl.Leg, wind: nl.Wind) -> float:
    """How long the row takes in a wind, or `_UNFLYABLE` if it cannot be flown."""
    if row.tas_kt <= 0.0:
        return 0.0
    try:
        triangle = solve_wind_triangle(
            row.true_course_deg, row.tas_kt, wind.from_deg, wind.speed_kt
        )
    except WindTooStrong:
        return _UNFLYABLE
    if triangle.ground_speed_kt <= 0.0:
        return _UNFLYABLE
    return 60.0 * row.distance_nm / triangle.ground_speed_kt


def _wind_over(rows: list[nl.Leg], winds: nl.WindsAloft) -> nl.Wind | None:
    """The wind at the altitude the leg is mostly flown at, for reporting."""
    if not rows:
        return None
    highest = max(rows, key=lambda row: row.distance_nm)
    return winds.at(highest.altitude_ft)


def _series(
    forecasts: tuple[PointForecast, ...], index: int
) -> PointForecast | None:
    """The series for one waypoint, or `None` where the list is short."""
    if 0 <= index < len(forecasts) and forecasts[index].hours:
        return forecasts[index]
    return None


def _arrival_times(
    navlog: nl.Navlog, drawn: list[nl.Waypoint], off_blocks: datetime | None
) -> list[datetime | None]:
    """When each waypoint the pilot drew is reached, one per waypoint.

    From the navlog's own cumulative times, which is the only place they
    exist. `None` throughout with no off-blocks time: every forecast is then
    read at the hour it was fetched for, and the loop settles on the first
    pass because nothing about the times can move it.
    """
    if off_blocks is None:
        return [None] * len(drawn)
    start = _utc(off_blocks)
    reached: dict[str, float] = {}
    for leg in navlog.legs:
        # The first arrival wins: a name that appears twice on a route is
        # reached the first time it is reached.
        if leg.to_name not in reached and leg.cumulative_ete_min is not None:
            reached[leg.to_name] = leg.cumulative_ete_min
    times = [start]
    for waypoint in drawn[1:]:
        minutes = reached.get(waypoint.name)
        times.append(None if minutes is None else start + timedelta(minutes=minutes))
    # A waypoint the navlog never arrived at leaves a hole; the hour before it
    # is the best estimate available and beats no answer at all.
    for index in range(1, len(times)):
        if times[index] is None:
            times[index] = times[index - 1]
    return times


def _same(left: tuple[LegWeather, ...], right: tuple[LegWeather, ...]) -> bool:
    """Whether two weather lists make the same choices."""
    return [entry.key for entry in left] == [entry.key for entry in right]


# --- handing the choice back to the navlog --------------------------------


def _with_weather(
    conditions: nl.Conditions,
    drawn: list[nl.Waypoint],
    weather: tuple[LegWeather, ...],
) -> nl.Conditions:
    """The conditions a pass is planned in, given the weather chosen for it.

    The column goes in whole -- wind and temperature together -- placed at
    each leg's midpoint, which is how `navlog._RouteColumns` finds it again:
    by where it is, so that the tops of climb can move under it without it
    moving with them. The navlog settles each leg's column into the one wind
    and one ISA deviation the leg is flown in (`navlog._segment_air`). The
    temperatures also go in route-wide, merging with the fields' own reports
    into the curve that legs with no column of their own, and the rows on the
    ground, still read.
    """
    if not weather:
        return conditions
    spans = nl.drawn_spans(drawn)
    columns = tuple(
        nl.WindColumn(
            spans[entry.leg].point_at_fraction(0.5), entry.winds, entry.temperatures
        )
        for entry in weather
        if entry.has_forecast and entry.leg < len(spans)
    )
    samples = tuple(
        sample for entry in weather for sample in entry.temperatures
    )
    return nl.Conditions(
        **{
            **conditions.__dict__,
            "wind_field": nl.WindField(columns) if columns else None,
            "temperatures_aloft": conditions.temperatures_aloft + samples,
        }
    )


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
