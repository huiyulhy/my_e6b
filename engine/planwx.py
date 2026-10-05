"""The weather list beside the navigation log, and the loop that settles them.

Two lists, one entry per leg the pilot drew: the navlog rows, and the weather
each of them was planned in. `solve` produces both, and they agree -- which is
the whole difficulty, because each is an input to the other.

**One forecast per leg, over its midpoint.** A leg is planned in a single
column of air, read by altitude (see `navlog._RouteColumns`), and the column is
fetched over the leg's midpoint -- the one point nearest, on average, to every
part of the leg. The model is asked there directly rather than at the two ends
and averaged, since an average of two columns is air that was forecast
nowhere. Legs are short enough for one column to stand for the whole of one.

It was once the other way: each leg costed under the forecasts at both of its
ends and planned in whichever burned more fuel. That biased every plan late,
and a flight that beats its plan on every leg is a plan that is not saying
what the day will be.

The column is read at the hour the aeroplane passes the leg's midpoint, which
is not known until the plan is built -- and the plan is built in the weather:

1. Plan with no forecast, to learn roughly when each leg is flown.
2. Read each leg's forecast at the hour its midpoint is passed.
3. Plan again on that weather, which moves the times -- and the tops of climb.
4. Re-read. If no leg's hour changed, the two lists agree and it is done.
"""

from __future__ import annotations

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

# Max allowable number of passes before the loop gives up settling
MAX_PASSES = 6


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
    """Every hour that was fetched over one leg's midpoint.

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
    """The weather one drawn leg was planned in.

    The second of the two lists. One entry per leg the pilot drew, in route
    order, including legs no forecast was available for -- a gap in the list
    would silently renumber it against the first.
    """

    leg: int  # index into the legs the pilot drew
    from_name: str
    to_name: str

    # The forecast hour read, or `None` where there was nothing to read.
    valid_time: datetime | None = None
    winds: nl.WindsAloft = dataclass_field(default_factory=nl.WindsAloft.calm)
    temperatures: tuple[TemperatureSample, ...] = ()
    # The forecast's wind at the altitude the leg is mostly flown at, for the
    # panel -- reported even when it could not be planned on.
    wind: nl.Wind | None = None
    # A wind no heading can hold the course in. Recorded rather than planned
    # on; see `_read_leg`.
    unflyable: bool = False
    note: str = ""

    @property
    def has_forecast(self) -> bool:
        """Whether the leg is planned on this forecast."""
        return self.valid_time is not None and not self.unflyable

    @property
    def key(self) -> tuple:
        """What has to stop changing before the two lists have settled.

        Which hour was read, and whether it could be planned on: both are
        discrete, so this either changes on a pass or it does not.
        """
        return (self.has_forecast, self.valid_time)


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
                    f"{self.passes} passes: the forecast hour each leg is read "
                    f"at keeps changing as the times move. The plan shown is "
                    f"the last pass -- check the wind on each row against the "
                    f"forecast you were briefed on"
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

    `forecasts` is one series per leg the pilot drew, fetched over the leg's
    midpoint, in route order -- one fewer than the route with its generated
    points removed. A short list, or one with empty series in it, is not an
    error: the legs it cannot cover are planned on whatever was typed on their
    rows, and on calm below that, which is what a plan with no weather on it
    has always meant.

    With no forecasts at all this is `build_navlog`, and it returns on the
    first pass.
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

    # Pass zero: no forecast at all, purely to learn roughly when each leg is
    # flown. Its own winds are whatever the pilot typed.
    weather: tuple[LegWeather, ...] = ()
    navlog = plan(weather)
    if not forecasts or not any(series.hours for series in forecasts):
        return SolvedPlan(navlog=navlog, weather=(), passes=1, settled=True)

    for attempt in range(1, max_passes + 1):
        read = _read(drawn, navlog, forecasts, off_blocks)
        if _same(read, weather):
            # The hours survived the plan they produced. The two lists agree.
            return SolvedPlan(
                navlog=navlog, weather=read, passes=attempt, settled=True
            )
        weather = read
        navlog = plan(weather)

    # Out of passes. The last plan is returned rather than nothing -- it is a
    # real plan on a real forecast, just not one that proved stable -- and
    # `SolvedPlan.notes` says so.
    return SolvedPlan(
        navlog=navlog, weather=weather, passes=max_passes, settled=False
    )


# --- reading --------------------------------------------------------------


def _read(
    drawn: list[nl.Waypoint],
    navlog: nl.Navlog,
    forecasts: tuple[PointForecast, ...],
    off_blocks: datetime | None,
) -> tuple[LegWeather, ...]:
    """The weather list for one pass, read off the navlog of the pass before.

    The navlog supplies two things: when each leg's midpoint is passed, which
    picks the forecast hour, and the leg's rows, which the forecast's wind is
    checked against.
    """
    times = _midpoint_times(navlog, drawn, off_blocks)
    rows = nl.rows_by_drawn_leg(navlog.legs, drawn)
    read: list[LegWeather] = []
    for index in range(len(drawn) - 1):
        series = _series(forecasts, index)
        read.append(
            _read_leg(
                index=index,
                from_name=drawn[index].name,
                to_name=drawn[index + 1].name,
                rows=rows.get(index, []),
                hour=None if series is None else series.at(times[index]),
            )
        )
    return tuple(read)


def _read_leg(
    *,
    index: int,
    from_name: str,
    to_name: str,
    rows: list[nl.Leg],
    hour: ForecastHour | None,
) -> LegWeather:
    """One leg, in the forecast over its midpoint.

    A wind no heading can hold the course in is not planned on. Planning on it
    would make the route unbuildable and leave the pilot looking at an error
    instead of at a plan -- so the leg falls back to the wind typed on its
    rows, and the fact is written on the leg, which says strictly more than
    the refusal would have.
    """
    identity = {"leg": index, "from_name": from_name, "to_name": to_name}
    if hour is None:
        return LegWeather(**identity, note="no forecast over this leg")

    found = {
        "valid_time": hour.valid_time,
        "wind": _wind_over(rows, hour.winds),
    }
    if not _flyable(rows, hour.winds):
        return LegWeather(
            **identity,
            **found,
            unflyable=True,
            note=(
                f"the forecast between {from_name} and {to_name} cannot be "
                f"flown at all -- the wind exceeds what the aeroplane can hold "
                f"the course against; this leg falls back to the wind typed on "
                f"its rows"
            ),
        )
    return LegWeather(
        **identity, **found, winds=hour.winds, temperatures=hour.temperatures
    )


def _flyable(rows: list[nl.Leg], winds: nl.WindsAloft) -> bool:
    """Whether every row of the leg can hold its course in this wind.

    Read off the rows as they already stand -- their course, altitude and true
    airspeed -- with only the wind varied. No rows, which is every leg on the
    first pass, has nothing to fail.
    """
    for row in rows:
        if row.tas_kt <= 0.0:
            continue
        wind = winds.at(row.altitude_ft)
        try:
            triangle = solve_wind_triangle(
                row.true_course_deg, row.tas_kt, wind.from_deg, wind.speed_kt
            )
        except WindTooStrong:
            return False
        if triangle.ground_speed_kt <= 0.0:
            return False
    return True


def _wind_over(rows: list[nl.Leg], winds: nl.WindsAloft) -> nl.Wind | None:
    """The wind at the altitude the leg is mostly flown at, for reporting."""
    if not rows:
        return None
    longest = max(rows, key=lambda row: row.distance_nm)
    return winds.at(longest.altitude_ft)


def _series(
    forecasts: tuple[PointForecast, ...], index: int
) -> PointForecast | None:
    """The series for one leg, or `None` where the list is short."""
    if 0 <= index < len(forecasts) and forecasts[index].hours:
        return forecasts[index]
    return None


def _midpoint_times(
    navlog: nl.Navlog, drawn: list[nl.Waypoint], off_blocks: datetime | None
) -> list[datetime | None]:
    """When each drawn leg's midpoint is passed, one per leg.

    Half way in time between reaching the leg's two ends. Not the same instant
    as half way along it -- the climb is slower over the ground than the cruise
    -- but the forecast is read to the nearest hour, and on a leg short enough
    for one column to stand for it the two round to the same one.
    """
    reached = _arrival_times(navlog, drawn, off_blocks)
    return [
        None if start is None or end is None else start + (end - start) / 2
        for start, end in zip(reached, reached[1:])
    ]


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
    """Whether two weather lists read the same hours."""
    return [entry.key for entry in left] == [entry.key for entry in right]


# --- handing the weather back to the navlog -------------------------------


def _with_weather(
    conditions: nl.Conditions,
    drawn: list[nl.Waypoint],
    weather: tuple[LegWeather, ...],
) -> nl.Conditions:
    """The conditions a pass is planned in, given the weather read for it.

    The column goes in whole -- wind and temperature together -- placed at
    the leg's midpoint, where it was fetched and which is how
    `navlog._RouteColumns` finds it again: by where it is, so that the tops of
    climb can move under it without it moving with them. The navlog flies each
    leg in its column's wind and temperature profiles, read at each row's
    altitude. The temperatures also go in route-wide, merging with the fields'
    own reports into the curve that legs with no column of their own, and the
    rows on the ground, still read.
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
