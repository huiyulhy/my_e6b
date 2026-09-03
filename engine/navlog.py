"""Generates final navlog

Turns a route -- an ordered list of waypoints plus a cruise altitude -- into
the table a pilot flies from: true course, variation, magnetic heading, wind
correction, ground speed, distance, time and fuel for every leg.

Types of flight segments
1. Taxi (start, taxi and takeoff allowance)
2. Climb
3. Cruise
4. Descent
5. Traffic pattern at the arrival field

All waypoints for start and end of each segment should be captured to cleanly
separate the flight segment types.

Three modelling decisions worth knowing, all of them visible in the output:
1. Top of climb and descent are considered waypoints on the navlog - so each 
segment only has 1 flight phase
2. Climb and descent distances come from flying the POH climb speed (or the
descent speed) against the forecast wind, not from the published no-wind
climb distance column. Time and fuel stay the published figures.
3. Descent is taken at a constant % of the cruise
4. Assume 10 mins in traffic pattern
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC
from datetime import date as Date
from datetime import datetime as DateTime
from itertools import pairwise

from engine import performance as perf
from engine import preflight
from engine.atmosphere import (
    TemperatureProfile,
    TemperatureSample,
    density_altitude,
    isa_temperature_c,
    pressure_altitude,
)
from engine.geo import (
    LatLon,
    Segment,
    WindTooStrong,
    inverse,
    solve_wind_triangle,
)
from engine.magnetic import (
    check_validity,
    decimal_year_for,
    true_to_magnetic,
    variation,
)
from engine.profile import (
    CONCRETE_SEGMENT_TYPES,
    PLANNING_MODES,
    PhaseNamer,
    ProfileSegment,
    RouteError,
    SegmentType,
    Waypoint,
    build_segments,
    resolve_route,
    strip_generated,
)

__all__ = [
    "CONCRETE_SEGMENT_TYPES",
    "PLANNING_MODES",
    "Aircraft",
    "Conditions",
    "Leg",
    "LegOverride",
    "Navlog",
    "PhaseNamer",
    "ProfileSegment",
    "RouteError",
    "SegmentType",
    "TemperatureProfile",
    "TemperatureSample",
    "TypedWind",
    "Waypoint",
    "Wind",
    "WindColumn",
    "WindField",
    "WindsAloft",
    "build_navlog",
    "build_segments",
    "format_navlog",
    "resolve_route",
    "strip_generated",
]

# FAR 91.151: Fuel reserve
DAY_VFR_RESERVE_MIN = 30.0
NIGHT_VFR_RESERVE_MIN = 45.0

# Floor for a per-row cruise power setting, as a pressure altitude. Below this
# the aeroplane is still in the departure or arrival environment
MIN_CUSTOM_CRUISE_RPM_PRESSURE_ALT_FT = 3000.0

# A leg shorter than this earns no row. Not just `== 0`: when a climb and a
# descent are scaled to fit a short leg their join lands on nearly the same
# point, leaving a row of no length whose course is numerical noise.
_MIN_ROW_NM = 0.1


# --- inputs --------------------------------------------------------------
# `Waypoint`, `RouteError` and the segment model live in `engine.profile`,
# which owns the vertical profile. They are re-exported here because this is
# the module callers build a plan through.


@dataclass(frozen=True)
class Wind:
    from_deg: float  # direction the wind blows FROM, true
    speed_kt: float


@dataclass(frozen=True)
class WindsAloft:
    """Forecast winds by altitude, as an FB product gives them.

    Interpolation is done on the north/east velocity component
    """

    layers: tuple[tuple[float, Wind], ...] = ()

    @classmethod
    def calm(cls) -> WindsAloft:
        return cls(())

    @classmethod
    def uniform(cls, from_deg: float, speed_kt: float) -> WindsAloft:
        """One wind at every altitude -- the usual case for a short flight."""
        return cls(((0.0, Wind(from_deg, speed_kt)),))

    def at(self, altitude_ft: float) -> Wind:
        if not self.layers:
            return Wind(0.0, 0.0)
        ordered = sorted(self.layers, key=lambda pair: pair[0])
        if len(ordered) == 1 or altitude_ft <= ordered[0][0]:
            return ordered[0][1]
        if altitude_ft >= ordered[-1][0]:
            return ordered[-1][1]

        for (low_alt, low), (high_alt, high) in pairwise(ordered):
            if low_alt <= altitude_ft <= high_alt:
                span = high_alt - low_alt
                fraction = 0.0 if span == 0 else (altitude_ft - low_alt) / span
                # Blend as vectors, pointing the way the wind travels.
                lu, lv = _wind_components(low)
                hu, hv = _wind_components(high)
                u = lu + fraction * (hu - lu)
                v = lv + fraction * (hv - lv)
                speed = math.hypot(u, v)
                if speed == 0.0:
                    return Wind(0.0, 0.0)
                return Wind((math.degrees(math.atan2(-u, -v))) % 360.0, speed)
        return ordered[-1][1]


def _wind_components(wind: Wind) -> tuple[float, float]:
    """East and north components of wind vector, in knots."""
    towards = math.radians((wind.from_deg + 180.0) % 360.0)
    return wind.speed_kt * math.sin(towards), wind.speed_kt * math.cos(towards)


@dataclass(frozen=True)
class WindColumn:
    """One forecast column of wind, and the point it was forecast over."""

    position: LatLon
    winds: WindsAloft


@dataclass(frozen=True)
class WindField:
    """Several columns of wind across a route, each over its own point.

    A `WindsAloft` is one column: it answers by altitude, and it is the whole
    sky as far as the plan is concerned. That is the FD product's shape, and
    the objection `engine/aloft.py` raises against it -- the wind on the coast
    is not the wind over the valley -- applies just as much to one gridded
    column stretched over a 300 nm route.

    So a leg is flown in the column nearest the ground it covers. Nearest
    rather than blended between the two: each column is fetched *for* a leg,
    so a leg's own column is the nearest one to its own midpoint by
    construction, and averaging two would smear across a front the model had
    resolved sharply.

    Keyed by position rather than by row or leg index on purpose. A forecast
    wind moves the tops of climb and descent, which renumbers the very rows it
    would have been keyed to; a point on the earth does not move. Which leg a
    column belongs to is worked out from where it is -- see `_RouteColumns`.
    """

    columns: tuple[WindColumn, ...] = ()


@dataclass(frozen=True)
class TypedWind:
    """One leg's wind, as far as the pilot typed it.

    Answers `at()` like `WindsAloft`, so it can stand in for the route's wind
    profile on the leg it belongs to -- but it is a leg's wind, not an
    altitude's: the same answer all the way up the climb and all the way along
    the ground the row covers, which is what "the wind on this leg" means.

    Either half may be missing, because either half may be typed on its own.
    What was not typed is read off the route's own profile at the altitude
    asked for, so half an entry never quietly zeroes the other half.
    """

    base: WindsAloft
    from_deg: float | None = None
    speed_kt: float | None = None

    def at(self, altitude_ft: float) -> Wind:
        under = self.base.at(altitude_ft)
        return Wind(
            under.from_deg if self.from_deg is None else self.from_deg % 360.0,
            under.speed_kt if self.speed_kt is None else self.speed_kt,
        )


@dataclass(frozen=True)
class Conditions:
    """Atmospheric conditions for the flight."""

    altimeter_inhg: float = 29.92126
    isa_deviation_c: float = 0.0
    winds: WindsAloft = dataclass_field(default_factory=WindsAloft.calm)
    flight_date: Date | None = None
    night: bool = False
    temperatures: TemperatureProfile | None = None
    temperatures_aloft: tuple[TemperatureSample, ...] = ()
    # Forecast columns across the route, when any were fetched. Where there
    # are none every leg reads `winds`, which is what a plan with nothing
    # entered has always meant.
    wind_field: WindField | None = None

    def oat_c(self, altitude_ft: float) -> float:
        """Temperature at an indicated altitude, observed where it was reported.

        Referenced to pressure altitude rather than to the indicated altitude,
        so we can get temperature deviation for cruise table interpolation
        """
        pa = self.pressure_altitude_ft(altitude_ft)
        deviation = (
            self.isa_deviation_c
            if self.temperatures is None
            else self.temperatures.deviation_at(pa)
        )
        return isa_temperature_c(pa) + deviation

    def pressure_altitude_ft(self, indicated_altitude_ft: float) -> float:
        return pressure_altitude(indicated_altitude_ft, self.altimeter_inhg)

    def density_altitude_ft(self, indicated_altitude_ft: float) -> float:
        return density_altitude(
            self.pressure_altitude_ft(indicated_altitude_ft),
            self.oat_c(indicated_altitude_ft),
        )

    @property
    def reserve_minutes(self) -> float:
        return NIGHT_VFR_RESERVE_MIN if self.night else DAY_VFR_RESERVE_MIN


# --- the air at a point on the ground ------------------------------------
@dataclass(frozen=True)
class FieldWeather:
    """The air at field conditions, from METARs/TAF
    The rule: the waypoint's own altimeter setting and temperature win wherever they were given, and
    the route-wide figures fill in where they were not.

    Three call sites used to spell that fallback out for themselves -- the
    go/no-go check, the taxi and pattern rows, and the temperature profile
    """

    # The altitude this describes: the altitude at which we are interested in
    altitude_ft: float
    altimeter_inhg: float
    oat_c: float
    pressure_altitude_ft: float
    density_altitude_ft: float

    @property
    def isa_deviation_c(self) -> float:
        return self.oat_c - isa_temperature_c(self.pressure_altitude_ft)


def field_weather(
    waypoint: Waypoint,
    conditions: Conditions,
    *,
    altitude_ft: float | None = None,
    require_reported_temperature: bool = False,
) -> FieldWeather | None:
    """Pressure and density altitude at a waypoint's own elevation.

    `None` when there is no elevation to read them at: an en-route fix over
    open country has no field, and inventing one would put a density altitude
    on the screen that means nothing.

    `altitude_ft` overrides the elevation for a row that happens above the
    field -- the traffic pattern -- while still reading that field's altimeter
    setting and temperature, which is what a pilot in the circuit has.

    `require_reported_temperature` refuses the route fallback rather than
    filling it in. The temperature profile is built from these, so a fallback
    there would feed the profile its own output back as an observation.
    """
    elevation = altitude_ft if altitude_ft is not None else waypoint.elevation_ft
    if elevation is None:
        return None
    if require_reported_temperature and waypoint.oat_c is None:
        return None

    altimeter = (
        waypoint.altimeter_inhg
        if waypoint.altimeter_inhg is not None
        else conditions.altimeter_inhg
    )
    oat = waypoint.oat_c if waypoint.oat_c is not None else conditions.oat_c(elevation)
    pressure_alt = pressure_altitude(elevation, altimeter)
    return FieldWeather(
        altitude_ft=elevation,
        altimeter_inhg=altimeter,
        oat_c=oat,
        pressure_altitude_ft=pressure_alt,
        density_altitude_ft=density_altitude(pressure_alt, oat),
    )


@dataclass(frozen=True)
class Aircraft:
    """The airplane and how it is being flown.

    Defaults describe a 172S at gross weight in an economical cruise.
    """

    weight_lb: float = perf.MAX_GROSS_WEIGHT_LB
    cruise_rpm: float = 2400.0
    fuel_on_board_gal: float = perf.FUEL_USABLE_GAL
    # POH allowance for engine start, taxi and takeoff, charged once per
    # departure rather than once per flight plan.
    taxi_fuel_gal: float = perf.START_TAXI_TAKEOFF_FUEL_GAL
    descent_rate_fpm: float = 500.0
    descent_speed_kias: float = 90.0
    # Time in the traffic pattern at each arrival, but assume cruise fuel flow
    pattern_time_min: float = 10.0
    # Pattern altitude above field elevation, for the row's altitude column.
    pattern_height_agl_ft: float = 1000.0


@dataclass(frozen=True)
class LegOverride:
    """Manual values for one navlog row, replacing what was computed.
    Captures the leg override made by manual edits
    
    Overriding a row recomputes everything *downstream* of the value, and it
    never changes which rows exist.

    `altitude_ft` replaces the altitude this leg ends at.

    `wind_from_deg` and `wind_speed_kt` are the wind on this leg, and they go
    in before the profile is solved rather than after it. A wind is a fact
    about a stretch of the route, so it holds for the whole leg -- all the way
    up a climb and along all the ground the row covers -- and the tops of climb
    and descent move under it: into a headwind the same climb covers less
    ground, so it tops out sooner along the route. Either half may be typed on
    its own; whatever is not typed is read off the route's own wind profile.
    See `TypedWind` and `profile.build_segments`.

    `cruise_rpm` re-reads the POH cruise chart for this row alone, replacing
    both its true airspeed and its fuel flow. Unlike the other fields it is
    not a free number: see `_leg_cruise_point` for the two conditions it has
    to meet. Setting `tas_kt` as well keeps the RPM's fuel flow but takes the
    airspeed from the pilot, which is how a leaned cruise off the POH's
    "recommended lean" column gets planned.

    `pressure_altitude_ft` replaces the pressure altitude this row's air is
    read at -- what the pilot gets off the altimeter with 29.92 set, rather
    than what the route's altimeter setting implies. It is the row's own
    number, not the air's: unlike `oat_c` it does not lapse into the rows
    around it. Density altitude follows from it and the row's temperature, and
    so does every chart this row reads at its own altitude -- the cruise chart
    that sets a level row's true airspeed and fuel flow, and the fuel flow a
    descent is charged at. It does **not** move the POH climb figures: time and
    fuel to climb are integrated over the whole climb before the rows exist,
    from the profile's altitudes. A row that says so gets a warning.

    `oat_c` is the temperature at this row's altitude, off an FD forecast. It
    is the one field that is not confined to its own row: a temperature is an
    observation about the *air*, so it joins the field METARs in the route's
    temperature profile and lapses from there into the rows around it. That
    also makes it the one field the plan has to be built twice for -- the
    altitude to hang the observation at only exists once the rows do. See
    `_build_flight`.
    """

    wind_from_deg: float | None = None
    wind_speed_kt: float | None = None
    tas_kt: float | None = None
    cruise_rpm: float | None = None
    altitude_ft: float | None = None
    oat_c: float | None = None
    pressure_altitude_ft: float | None = None

    @property
    def is_empty(self) -> bool:
        return (
            self.wind_from_deg is None
            and self.wind_speed_kt is None
            and self.tas_kt is None
            and self.cruise_rpm is None
            and self.altitude_ft is None
            and self.oat_c is None
            and self.pressure_altitude_ft is None
        )


# --- outputs -------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One row of the navigation log."""

    from_name: str
    to_name: str
    phase: str  # taxi | climb | cruise | descent | pattern
    altitude_ft: float

    # Endpoint positions. Carried on the leg because top of climb and top of
    # descent are computed here and exist nowhere in the caller's route, so
    # this is the only way a map can draw the phases where they actually are.
    from_position: LatLon
    to_position: LatLon

    true_course_deg: float
    variation_deg: float  # positive east
    magnetic_course_deg: float
    wind_correction_angle_deg: float
    true_heading_deg: float
    magnetic_heading_deg: float

    wind_from_deg: float
    wind_speed_kt: float
    headwind_kt: float

    tas_kt: float
    ground_speed_kt: float

    distance_nm: float
    cumulative_distance_nm: float
    ete_min: float
    cumulative_ete_min: float
    fuel_gal: float
    cumulative_fuel_gal: float
    fuel_remaining_gal: float

    # Which fields on this row came from a manual override rather than the
    # model. The UI marks them so an edited row never passes for a computed
    # one, and `format_navlog` flags them in the printed log.
    overridden: tuple[str, ...] = ()

    # Which flight of a multi-stop route this row belongs to, counting from
    # zero. Every intermediate landing starts a new one.
    flight_index: int = 0

    # The cruise power setting this row's airspeed and fuel flow were read at.
    # None on rows that are not level flight, which have no cruise setting.
    cruise_rpm: float | None = None

    # What this leg does about altitude: climb | cruise | descent. The declared
    # type, which is not always what the altitudes ended up doing -- see
    # `engine.consistency`. None on the ground rows.
    segment_type: str | None = None

    # Altitudes at the ends of the leg, as opposed to `altitude_ft`, which is
    # the representative midpoint the performance was read at.
    entry_altitude_ft: float | None = None
    exit_altitude_ft: float | None = None

    # The air this row was flown through, at `altitude_ft`. Reported because
    # every performance number on the row was read at this density altitude,
    # and a pilot checking the plan against a chart needs to see which one.
    oat_c: float | None = None
    pressure_altitude_ft: float | None = None
    density_altitude_ft: float | None = None

    # "TOC" / "TOD" when this row starts or ends at a top of climb or descent.
    # Carried beside the name rather than replacing it, so a charted waypoint
    # used as a top of climb still prints the name on the sectional.
    start_role: str | None = None
    end_role: str | None = None

    @property
    def covers_ground(self) -> bool:
        """Whether this row is a leg flown along the route.

        False for taxi and the pattern buffer, which burn time and fuel at a
        point rather than between two of them. Their course, heading, wind and
        airspeed columns are meaningless and every renderer blanks them out.
        """
        return self.phase not in ("taxi", "pattern")


@dataclass(frozen=True)
class Navlog:
    legs: tuple[Leg, ...]
    total_distance_nm: float
    total_time_min: float
    total_fuel_gal: float  # includes the taxi allowance
    fuel_remaining_gal: float
    reserve_required_gal: float
    cruise_altitude_ft: float
    warnings: tuple[str, ...]

    # Which mode built this: "manual" | "auto". See `build_navlog`.
    planning_mode: str = "manual"

    # The route the rows were actually built from. In automatic mode this has
    # the TOC/TOD points the planner inserted, which the caller's route did
    # not, so handing it back is what lets the pilot see and edit them.
    resolved_waypoints: tuple[Waypoint, ...] = ()

    # The go/no-go checklist for this route: every takeoff and landing against
    # the runway available, and the fuel against the reserve.
    checklist: preflight.GoNoGo | None = None

    # The conditions the rows were actually computed at, which is not quite the
    # object the caller passed in: the temperature profile is folded in from
    # everything reported along the route before anything is computed. Handed
    # back so a caller asking "what was the air at this field?" gets the same
    # answer the log did -- see `field_weather`.
    conditions: Conditions | None = None

    @property
    def is_legal_on_fuel(self) -> bool:
        """Whether the flight lands with its required VFR reserve intact."""
        return self.fuel_remaining_gal >= self.reserve_required_gal


# --- route geometry ------------------------------------------------------


def _leg_distances(waypoints: list[Waypoint]) -> list[float]:
    return [
        inverse(a.position, b.position).distance_nm
        for a, b in pairwise(waypoints)
    ]


# --- the calculation -----------------------------------------------------


def build_navlog(
    waypoints: list[Waypoint],
    cruise_altitude_ft: float,
    aircraft: Aircraft | None = None,
    conditions: Conditions | None = None,
    overrides: dict[int, LegOverride] | None = None,
    margins: preflight.Margins | None = None,
    planning_mode: str = "manual",
) -> Navlog:
    """Compute a full navigation log for a route.

    `waypoints` must start at the departure airport and end at the
    destination; both need an `elevation_ft` so climb and descent can be
    computed.

    `planning_mode` picks who decides the vertical profile:

    * `"manual"` -- the default. The pilot declares every leg's `segment_type`
      and the altitudes follow from performance. `cruise_altitude_ft` is then
      only a sanity bound, not a target. This is the mode a pilot plans in:
      the profile is the one they chose, not one inferred from an altitude.
    * `"auto"` -- the pilot names a `cruise_altitude_ft` and the planner works
      out where the climb tops out and the descent begins, inserting TOC and
      TOD into the route.

    Either way the route is *resolved* into one segment per leg before any row
    is built, so `overrides` -- which map a row index to manual values -- line
    up 1:1 with legs.
    """
    aircraft = aircraft or Aircraft()
    conditions = conditions or Conditions()
    margins = margins or preflight.Margins()
    overrides = {i: o for i, o in (overrides or {}).items() if not o.is_empty}
    warnings: list[str] = []

    if planning_mode not in PLANNING_MODES:
        raise RouteError(
            f"unknown planning mode {planning_mode!r}; "
            f"expected one of {', '.join(PLANNING_MODES)}"
        )
    if len(waypoints) < 2:
        raise RouteError("a route needs at least a departure and a destination")

    # Planner-inserted points from a previous run are dropped before anything
    # else, so re-planning is idempotent rather than compounding.
    waypoints = strip_generated(waypoints)
    flights = _split_at_landings(waypoints)
    # Validated before any performance lookup, so a route that cannot be flown
    # says why rather than failing later as "no published cruise data".
    for flight in flights:
        _require_flyable_altitudes(flight, cruise_altitude_ft, planning_mode)

    # Every field temperature reported anywhere on the route, folded into one
    # curve. Each flight narrows this to its own fields once it is built, but
    # the figures below are for the day as a whole, and an airport that
    # reported nothing is better served by its neighbours' air than by a form
    # default nobody revisited.
    conditions = _with_temperatures(
        conditions, _observed_samples(waypoints, conditions)
    )

    # The flight's nominal cruise point: the chart read at the altitude the
    # pilot named. Level rows read the chart at their own altitude instead, and
    # the pattern and reserve read it where they are actually flown, so this is
    # only a fallback for a profile that has no level flight in it to measure.
    # Clamped into the chart because in user-driven mode the cruise altitude is
    # only a bound and may sit below its 2000 ft floor -- refusing to plan a low
    # route over a fallback estimate would be absurd.
    cruise_point = _cruise_point_at(
        cruise_altitude_ft,
        aircraft=aircraft,
        conditions=conditions,
        default=perf.cruise(
            perf.cruise_altitude_range()[0],
            aircraft.cruise_rpm,
            isa_temperature_c(perf.cruise_altitude_range()[0]),
        ),
    )

    # Today, not a fixed date: an unplanned flight is a flight now, and the
    # variation a fixed default gives drifts further out of true every year.
    flight_date = conditions.flight_date or DateTime.now(tz=UTC).date()
    decimal_year = decimal_year_for(flight_date.year, flight_date.month, flight_date.day)
    # Fail before any legs are built: an expired magnetic model makes every
    # magnetic course on the log wrong, so there is nothing worth computing.
    check_validity(decimal_year)

    legs: list[Leg] = []
    travelled = 0.0
    cumulative_time = 0.0
    # Charged once per takeoff: each departure taxis out on its own.
    cumulative_fuel = 0.0
    # One namer for the whole route, so a multi-stop day numbers its tops of
    # climb straight through instead of restarting at every landing.
    names = PhaseNamer()
    # The route after resolution, with any TOC/TOD the planner inserted. Given
    # back to the caller so the pilot can see and edit the points the planner
    # chose, rather than them existing only inside the finished log.
    resolved: list[Waypoint] = []

    for flight_index, flight in enumerate(flights):
        # The taxi row goes in before the flight is built so that the row
        # indices an override is keyed by keep matching the finished log.
        if aircraft.taxi_fuel_gal > 0:
            taxi_fuel = _fuel_written_on_the_log(aircraft.taxi_fuel_gal)
            cumulative_fuel += taxi_fuel
            legs.append(
                _ground_leg(
                    phase="taxi",
                    waypoint=flight[0],
                    altitude_ft=flight[0].elevation_ft or 0.0,
                    ete_min=0.0,
                    fuel_gal=taxi_fuel,
                    aircraft=aircraft,
                    conditions=conditions,
                    travelled=travelled,
                    cumulative_time=cumulative_time,
                    cumulative_fuel=cumulative_fuel,
                    flight_index=flight_index,
                )
            )

        (
            flight_legs,
            flight_route,
            travelled,
            cumulative_time,
            cumulative_fuel,
        ) = _build_flight(
            flight,
            flight_index=flight_index,
            cruise_altitude_ft=cruise_altitude_ft,
            aircraft=aircraft,
            conditions=conditions,
            cruise_point=cruise_point,
            names=names,
            overrides=overrides,
            row_offset=len(legs),
            decimal_year=decimal_year,
            travelled=travelled,
            cumulative_time=cumulative_time,
            cumulative_fuel=cumulative_fuel,
            warnings=warnings,
            planning_mode=planning_mode,
        )
        legs.extend(flight_legs)
        # A landing airport ends one flight and starts the next, so it is
        # already the last point of the previous one.
        resolved.extend(flight_route[1:] if resolved else flight_route)

        # Every arrival flies a pattern, including the ones at intermediate
        # stops, so the reserve check below sees the fuel it costs.
        if aircraft.pattern_time_min > 0:
            pattern_altitude = (
                flight[-1].elevation_ft or 0.0
            ) + aircraft.pattern_height_agl_ft
            # Charged where the pattern is actually flown, not at the altitude
            # the flight cruised at. A circuit at 1,100 ft costs what a circuit
            # at 1,100 ft costs, whether the trip there was at 3,500 or 9,500 --
            # and it costs *more*, because the engine makes more power low down.
            pattern_fuel = _fuel_written_on_the_log(
                _cruise_point_at(
                    pattern_altitude,
                    aircraft=aircraft,
                    conditions=conditions,
                    default=cruise_point,
                ).gph
                * aircraft.pattern_time_min
                / 60.0
            )
            cumulative_time += aircraft.pattern_time_min
            cumulative_fuel += pattern_fuel
            legs.append(
                _ground_leg(
                    phase="pattern",
                    waypoint=flight[-1],
                    altitude_ft=pattern_altitude,
                    ete_min=aircraft.pattern_time_min,
                    fuel_gal=pattern_fuel,
                    aircraft=aircraft,
                    conditions=conditions,
                    travelled=travelled,
                    cumulative_time=cumulative_time,
                    cumulative_fuel=cumulative_fuel,
                    flight_index=flight_index,
                )
            )

        # Every landing must arrive with its reserve intact, not just the last
        # one -- a stop that lands on fumes is illegal even if the next tank
        # of fuel is waiting on the ramp.
        if flight_legs and flight[-1] is not waypoints[-1]:
            arriving_with = aircraft.fuel_on_board_gal - cumulative_fuel
            leg_reserve = (
                _reserve_gph(legs, cruise_point.gph)
                * conditions.reserve_minutes
                / 60.0
            )
            if arriving_with < leg_reserve:
                warnings.append(
                    f"lands at {flight[-1].name} with {arriving_with:.1f} gal, "
                    f"below the {conditions.reserve_minutes:.0f}-minute VFR "
                    f"reserve of {leg_reserve:.1f} gal (FAR 91.151). Fuel is "
                    f"not assumed to be uplifted at a stop."
                )

    return _finish_navlog(
        legs=legs,
        overrides=overrides,
        aircraft=aircraft,
        conditions=conditions,
        cruise_point=cruise_point,
        cruise_altitude_ft=cruise_altitude_ft,
        planning_mode=planning_mode,
        resolved_waypoints=resolved,
        travelled=travelled,
        cumulative_time=cumulative_time,
        cumulative_fuel=cumulative_fuel,
        warnings=warnings,
        checklist=_build_checklist(
            flights,
            aircraft,
            conditions,
            _reserve_gph(legs, cruise_point.gph),
            cumulative_fuel,
            margins,
            decimal_year,
        ),
    )


def _ground_leg(
    *,
    phase: str,
    waypoint: Waypoint,
    altitude_ft: float,
    ete_min: float,
    fuel_gal: float,
    aircraft: Aircraft,
    conditions: Conditions,
    travelled: float,
    cumulative_time: float,
    cumulative_fuel: float,
    flight_index: int,
) -> Leg:
    """A row that burns time and fuel at one point rather than between two.

    Taxi and the traffic pattern both work this way: no distance is covered,
    so course, heading, wind and airspeed are all left at zero and every
    renderer blanks them out rather than printing a heading nobody flies. The
    cumulative columns still run through the row, which is the whole point --
    the fuel a pilot reads at the bottom of the log is now the sum of the
    column above it.

    The air is still reported: this row happens at an airport, so its density
    altitude is the field's, read off that station's own METAR where one was
    given. It is the number the takeoff distance was computed at.
    """
    air = field_weather(waypoint, conditions, altitude_ft=altitude_ft)
    assert air is not None  # an explicit altitude is always readable
    return Leg(
        from_name=waypoint.name,
        to_name=waypoint.name,
        phase=phase,
        altitude_ft=altitude_ft,
        from_position=waypoint.position,
        to_position=waypoint.position,
        true_course_deg=0.0,
        variation_deg=0.0,
        magnetic_course_deg=0.0,
        wind_correction_angle_deg=0.0,
        true_heading_deg=0.0,
        magnetic_heading_deg=0.0,
        wind_from_deg=0.0,
        wind_speed_kt=0.0,
        headwind_kt=0.0,
        tas_kt=0.0,
        ground_speed_kt=0.0,
        distance_nm=0.0,
        cumulative_distance_nm=travelled,
        ete_min=ete_min,
        cumulative_ete_min=cumulative_time,
        fuel_gal=fuel_gal,
        cumulative_fuel_gal=cumulative_fuel,
        fuel_remaining_gal=aircraft.fuel_on_board_gal - cumulative_fuel,
        flight_index=flight_index,
        oat_c=air.oat_c,
        pressure_altitude_ft=air.pressure_altitude_ft,
        density_altitude_ft=air.density_altitude_ft,
    )


def _field_wind(waypoint: Waypoint, decimal_year: float) -> preflight.SurfaceWind | None:
    """The field's surface wind, turned from true into magnetic.

    A reported wind is true; a runway designator is magnetic. Comparing them
    without converting is wrong by the local variation, which is 13 degrees on
    the US west coast -- a quarter of the way to the next runway designator,
    and enough to move a crosswind across the demonstrated limit. `None` when
    the field has no wind on it, which the checklist reports rather than
    silently reading as calm.
    """
    if waypoint.wind_speed_kt is None:
        return None
    # A calm wind has no direction to convert, and needs none. Any gust goes
    # with it: a peak with no direction cannot be resolved onto a runway, and
    # inventing one to resolve it against would be worse than dropping it.
    if waypoint.wind_speed_kt == 0:
        return preflight.SurfaceWind(from_deg=0.0, speed_kt=0.0)
    if waypoint.wind_from_deg is None:
        return None
    var = variation(
        waypoint.position.lat,
        waypoint.position.lon,
        waypoint.elevation_ft or 0.0,
        decimal_year,
    )
    gust = waypoint.gust_kt
    return preflight.SurfaceWind(
        from_deg=true_to_magnetic(waypoint.wind_from_deg, var),
        speed_kt=waypoint.wind_speed_kt,
        # A gust at or below the steady wind is not a gust; dropping it beats
        # refusing the whole plan over a rounding in the report.
        gust_kt=gust if gust is not None and gust > waypoint.wind_speed_kt else None,
    )


def _build_checklist(
    flights: list[list[Waypoint]],
    aircraft: Aircraft,
    conditions: Conditions,
    reserve_gph: float,
    burn_gal: float,
    margins: preflight.Margins,
    decimal_year: float,
) -> preflight.GoNoGo:
    """Every takeoff and landing on the route, plus the fuel check.

    An intermediate stop is checked twice -- once landing in, once taking off
    again -- because the runway that is long enough to get into is not always
    long enough to get out of on a hot afternoon. The weather is checked at
    both ends for the same reason: it is checked at the time the field is
    used, and a stop landed into at noon is departed from at four.
    """
    checks: list[preflight.AirportCheck] = []
    seen: set[tuple[str, str]] = set()

    for flight in flights:
        for waypoint, operation in ((flight[0], "takeoff"), (flight[-1], "landing")):
            if waypoint.elevation_ft is None:
                continue
            key = (waypoint.name, operation)
            if key in seen:
                continue
            seen.add(key)

            # Field weather beats the en-route figures when it is given: the
            # density altitude these distances are read at is the one at the
            # airport, not the one along the way.
            air = field_weather(waypoint, conditions)
            assert air is not None  # the elevation was checked above
            checks.append(
                preflight.check_airport(
                    airport=waypoint.name,
                    operation=operation,
                    runways=waypoint.runways,
                    elevation_ft=air.altitude_ft,
                    oat_c=air.oat_c,
                    pressure_altitude_ft=air.pressure_altitude_ft,
                    density_altitude_ft=air.density_altitude_ft,
                    weight_lb=aircraft.weight_lb,
                    margin=margins.runway,
                    wind=_field_wind(waypoint, decimal_year),
                    # Only what the field actually reported. A waypoint with
                    # no weather on it leaves the VFR check out rather than
                    # failing it, so a plan built without fetching weather
                    # reads exactly as it did before.
                    weather=waypoint.field_weather_report,
                    pattern_altitude_agl_ft=(
                        waypoint.pattern_altitude_agl_ft
                        if waypoint.pattern_altitude_agl_ft is not None
                        else preflight.DEFAULT_PATTERN_HEIGHT_AGL_FT
                    ),
                )
            )

    fuel = preflight.check_fuel(
        fuel_on_board_gal=aircraft.fuel_on_board_gal,
        burn_gal=burn_gal,
        reserve_required_gal=_fuel_written_on_the_log(
            reserve_gph * conditions.reserve_minutes / 60.0
        ),
        reserve_minutes=conditions.reserve_minutes,
        margin=margins.fuel,
        night=conditions.night,
    )
    return preflight.summarise(checks, fuel)


def _split_at_landings(waypoints: list[Waypoint]) -> list[list[Waypoint]]:
    """Break a route into one list of waypoints per flight.

    A waypoint marked `is_landing` ends one flight and begins the next, so it
    appears as the last point of one sub-route and the first of the following
    one. The result is that each flight gets its own climb and descent.
    """
    flights: list[list[Waypoint]] = []
    current: list[Waypoint] = [waypoints[0]]
    for waypoint in waypoints[1:]:
        current.append(waypoint)
        if waypoint.is_landing and waypoint is not waypoints[-1]:
            flights.append(current)
            current = [waypoint]
    flights.append(current)

    for flight in flights:
        if len(flight) < 2:
            raise RouteError(
                f"{flight[0].name} is marked as a landing but has nothing after "
                f"it to fly to"
            )
    return flights


def _require_flyable_altitudes(
    waypoints: list[Waypoint], cruise_altitude_ft: float, planning_mode: str = "manual"
) -> tuple[float, float]:
    """Check one flight's endpoints and return their elevations.

    The cruise altitude only has to clear the fields in automatic mode, where
    it is what the planner aims for. In user-driven mode the altitudes come
    from the declared segments instead, so it is not a target and a route flown
    entirely low is perfectly legitimate.
    """
    departure, destination = waypoints[0], waypoints[-1]
    departure_elevation = departure.elevation_ft
    destination_elevation = destination.elevation_ft
    if departure_elevation is None or destination_elevation is None:
        raise RouteError(
            f"{departure.name} and {destination.name} need elevation_ft to "
            f"compute climb and descent"
        )
    if planning_mode == "auto" and cruise_altitude_ft <= max(
        departure_elevation, destination_elevation
    ):
        raise RouteError(
            f"cruise altitude {cruise_altitude_ft:g} ft is not above both "
            f"{departure.name} and {destination.name} "
            f"({departure_elevation:g} / {destination_elevation:g} ft)"
        )
    return departure_elevation, destination_elevation


def _build_flight(
    waypoints: list[Waypoint],
    *,
    flight_index: int,
    cruise_altitude_ft: float,
    aircraft: Aircraft,
    conditions: Conditions,
    cruise_point: perf.CruisePoint,
    names: PhaseNamer,
    overrides: dict[int, LegOverride],
    row_offset: int,
    decimal_year: float,
    travelled: float,
    cumulative_time: float,
    cumulative_fuel: float,
    warnings: list[str],
    planning_mode: str = "manual",
) -> tuple[list[Leg], list[Waypoint], float, float, float]:
    """One departure-to-landing flight: climb, cruise, descent.

    Returns the rows *and the resolved route they were built from*, since in
    automatic mode that route has TOC/TOD points in it the caller did not send.

    Distance, time and fuel carry in and out so a multi-stop route's
    cumulative columns run continuously across the whole day rather than
    resetting at each stop.
    """
    departure, destination = waypoints[0], waypoints[-1]
    departure_elevation, destination_elevation = _require_flyable_altitudes(
        waypoints, cruise_altitude_ft, planning_mode
    )

    if sum(_leg_distances(waypoints)) <= 0:
        raise RouteError(
            f"the flight from {departure.name} to {destination.name} has no length"
        )

    def lay_out(
        conditions: Conditions,
        names: PhaseNamer,
        warnings: list[str],
        drawn_leg_winds: dict[int, dict[str, TypedWind]] | None = None,
    ) -> tuple[list[Waypoint], list[ProfileSegment]]:
        """Resolve the route and cut it into one segment per leg.

        Resolve first -- every leg ends up with a concrete segment type, and in
        automatic mode the TOC/TOD points become real waypoints -- then build
        one segment per leg. Both modes share the second half.

        Typed winds go in twice, keyed two ways, because the two halves walk
        different routes: the planner walks the legs the pilot drew, and the
        segment builder walks the resolved route the planner produced.
        """
        route = resolve_route(
            waypoints,
            mode=planning_mode,
            cruise_altitude_ft=cruise_altitude_ft,
            departure_elevation=departure_elevation,
            destination_elevation=destination_elevation,
            aircraft=aircraft,
            conditions=conditions,
            names=names,
            warnings=warnings,
            leg_winds=drawn_leg_winds,
        )
        return route, build_segments(
            route,
            departure_elevation=departure_elevation,
            aircraft=aircraft,
            conditions=conditions,
            cruise_point=cruise_point,
            altitude_overrides=_altitude_overrides_by_leg(route, overrides, row_offset),
            leg_winds=_leg_winds_by_leg(
                route, row_winds, row_offset, conditions, columns
            ),
        )

    # --- the airmass this flight is planned in ---------------------------
    #
    # Field weather is known before anything is laid out. A temperature typed
    # against a navlog row is not: the altitude to hang it at is a segment
    # midpoint, which only exists once the profile has been built. So when
    # there are any, the flight is laid out twice -- once on the field
    # observations alone to learn what altitude each row flies at, then again
    # with the pilot's temperatures folded in at those altitudes.
    #
    # Twice, and no more. The second pass moves the tops of climb a little,
    # which would move the altitudes, which would move the samples; there is no
    # reason to think chasing that converges anywhere better than where it
    # started. The observations stay anchored to the altitudes the pilot saw
    # when they typed them, which is also the only version they can check.
    field_samples = _observed_samples(waypoints, conditions)
    conditions = _with_temperatures(conditions, field_samples)

    row_temperatures = {
        row: override.oat_c
        for row, override in overrides.items()
        if override.oat_c is not None
    }
    row_winds = _row_winds(overrides)
    # A typed wind needs the draft for the same reason a typed temperature
    # does, but only in automatic mode: there the planner decides where the
    # top of climb falls before any row exists, so a wind typed against a row
    # has to be traced back to the leg the pilot drew it on. In user-driven
    # mode the rows are the pilot's own legs and `_leg_winds_by_leg` re-keys
    # them without a rehearsal.
    #
    # The forecast needs no rehearsal either way: a column belongs to a leg by
    # where it was forecast, so it can be handed to the planner before
    # anything has been laid out.
    columns = _RouteColumns.build(waypoints, conditions.wind_field)
    drawn_leg_winds: dict[int, dict[str, object]] | None = (
        _forecast_winds_by_drawn_leg(columns) or None
    )
    draft_segments: list[ProfileSegment] | None = None
    if row_temperatures or (row_winds and planning_mode == "auto"):
        # A throwaway namer and warning list: this pass is scaffolding, and its
        # TOC numbering and warnings would otherwise be emitted twice.
        _, draft_segments = lay_out(conditions, PhaseNamer(), [])
        if row_temperatures:
            conditions = _with_temperatures(
                conditions,
                field_samples
                + _row_temperature_samples(
                    draft_segments, row_temperatures, row_offset, conditions
                ),
            )
        if row_winds and planning_mode == "auto":
            drawn_leg_winds = _leg_winds_by_drawn_leg(
                waypoints, draft_segments, row_winds, row_offset, conditions, columns
            )

    if conditions.temperatures is not None:
        warnings.extend(conditions.temperatures.lapse_warnings())

    route, segments = lay_out(conditions, names, warnings, drawn_leg_winds)

    if draft_segments is not None and _row_count(draft_segments) != _row_count(segments):
        # A leg crossed the tenth-of-a-mile threshold as the tops of climb
        # moved, so the rows renumbered under the values that were typed
        # against them. Rare, but silently attributing a pilot's number to a
        # different leg is not something to let pass.
        warnings.append(
            "the rows renumbered when the entered temperatures and winds were "
            "applied; check that each one is still on the leg you meant"
        )

    # --- walk the profile, one row per segment --------------------------
    legs: list[Leg] = []

    for segment in segments:
        start, end = segment.start, segment.end
        leg_geo = inverse(start.position, end.position)
        # Not just `== 0`: when a climb and a descent are scaled to fit a
        # short leg, their join lands on nearly the same point and leaves a
        # row of no length whose course is numerical noise. Below a tenth of
        # a mile there is nothing to fly and nothing worth printing.
        if leg_geo.distance_nm < _MIN_ROW_NM:
            continue

        phase = segment.phase
        altitude = segment.altitude_ft
        tas = segment.tas_kt

        # The wind over the ground this row covers: the column over the leg
        # it flies along where one was fetched, the route's single profile
        # where none was.
        column = None if columns is None else columns.over(start, end)
        wind = (column or conditions.winds).at(altitude)
        # The index this row takes in the finished navlog: rows already
        # emitted by earlier flights, plus rows emitted by this one. Zero
        # length legs are skipped above, so it matches what the caller sees.
        override = overrides.get(row_offset + len(legs))
        leg_name = f"{start.name} to {end.name}"

        # The air this row was flown through, settled before anything is read
        # at it. A typed temperature is reported verbatim rather than read back
        # off the profile: merging and the second pass can move it by a
        # fraction of a degree, and a number the pilot typed should read back
        # as the number they typed -- and be the number the charts are read at.
        leg_pressure_alt, pressure_offset = _leg_pressure_altitude(
            override, altitude, conditions, phase=phase,
            leg_name=leg_name, warnings=warnings,
        )
        leg_oat = (
            override.oat_c
            if override is not None and override.oat_c is not None
            else conditions.oat_c(altitude)
        )

        # A level row reads the cruise chart at its own altitude; a climb or a
        # descent has no cruise setting and keeps the flight's, which is only
        # used for its fuel flow.
        leg_cruise = _leg_cruise_point(
            override,
            phase=phase,
            pressure_altitude_ft=leg_pressure_alt,
            oat_c=leg_oat,
            entry_altitude_ft=segment.entry_altitude_ft,
            pressure_offset_ft=pressure_offset,
            conditions=conditions,
            aircraft=aircraft,
            default=cruise_point,
            leg_name=leg_name,
            warnings=warnings,
        )
        if phase == "cruise":
            tas = leg_cruise.ktas
            leg_rpm = (
                override.cruise_rpm
                if override is not None and override.cruise_rpm is not None
                else aircraft.cruise_rpm
            )
        else:
            leg_rpm = None

        wind, tas, overridden = _apply_override(override, wind, tas, leg_name)

        try:
            triangle = solve_wind_triangle(
                leg_geo.true_course_deg, tas, wind.from_deg, wind.speed_kt
            )
        except WindTooStrong as exc:
            raise RouteError(f"leg {leg_name}: {exc}") from exc

        if triangle.ground_speed_kt <= 0:
            raise RouteError(
                f"leg {start.name} to {end.name}: the headwind of "
                f"{triangle.headwind_kt:.0f} kt equals or exceeds true airspeed, "
                f"so the airplane makes no progress"
            )

        var = variation(
            start.position.lat, start.position.lon, altitude, decimal_year
        )

        ete = 60.0 * leg_geo.distance_nm / triangle.ground_speed_kt
        level_gph = (
            _worst_level_gph(
                segment.exit_altitude_ft,
                conditions=conditions,
                pressure_offset_ft=pressure_offset,
                default=leg_cruise.gph,
            )
            if segment.level_minutes > 0
            else None
        )
        fuel = _fuel_written_on_the_log(
            _leg_fuel(
                phase,
                ete,
                segment.climb,
                leg_cruise.gph,
                level_minutes=segment.level_minutes,
                level_gph=level_gph,
            )
        )

        travelled += leg_geo.distance_nm
        cumulative_time += ete
        cumulative_fuel += fuel

        legs.append(
            Leg(
                from_name=start.name,
                to_name=end.name,
                phase=phase,
                altitude_ft=altitude,
                from_position=start.position,
                to_position=end.position,
                true_course_deg=leg_geo.true_course_deg,
                variation_deg=var,
                magnetic_course_deg=true_to_magnetic(leg_geo.true_course_deg, var),
                wind_correction_angle_deg=triangle.wind_correction_angle_deg,
                true_heading_deg=triangle.true_heading_deg,
                magnetic_heading_deg=true_to_magnetic(triangle.true_heading_deg, var),
                wind_from_deg=wind.from_deg,
                wind_speed_kt=wind.speed_kt,
                headwind_kt=triangle.headwind_kt,
                tas_kt=tas,
                ground_speed_kt=triangle.ground_speed_kt,
                distance_nm=leg_geo.distance_nm,
                cumulative_distance_nm=travelled,
                ete_min=ete,
                cumulative_ete_min=cumulative_time,
                fuel_gal=fuel,
                cumulative_fuel_gal=cumulative_fuel,
                fuel_remaining_gal=aircraft.fuel_on_board_gal - cumulative_fuel,
                overridden=overridden,
                flight_index=flight_index,
                cruise_rpm=leg_rpm,
                segment_type=phase,
                entry_altitude_ft=segment.entry_altitude_ft,
                exit_altitude_ft=segment.exit_altitude_ft,
                oat_c=leg_oat,
                pressure_altitude_ft=leg_pressure_alt,
                density_altitude_ft=density_altitude(leg_pressure_alt, leg_oat),
                start_role=segment.start_role,
                end_role=segment.end_role,
            )
        )

    return legs, route, travelled, cumulative_time, cumulative_fuel


def _with_temperatures(
    conditions: Conditions, samples: list[TemperatureSample]
) -> Conditions:
    """Attach a temperature profile built from what was observed.

    The route-wide ISA deviation stays on as the fallback the profile uses
    where nothing was observed at all, so a plan with no weather entered is
    unaffected by any of this.
    """
    return replace(
        conditions,
        temperatures=TemperatureProfile.from_observations(
            samples, default_deviation_c=conditions.isa_deviation_c
        ),
    )


def _observed_samples(
    waypoints: list[Waypoint], conditions: Conditions
) -> list[TemperatureSample]:
    """Everything reported for this flight: the fields, and the FD levels.

    The two arrive in different coordinates and are reconciled here. A field
    temperature is read at a true elevation under that station's altimeter
    setting, so it needs converting. An FD level is already a pressure
    altitude, so it does not.
    """
    return _field_temperature_samples(waypoints, conditions) + list(
        conditions.temperatures_aloft
    )


def _field_temperature_samples(
    waypoints: list[Waypoint], conditions: Conditions
) -> list[TemperatureSample]:
    """Every field temperature reported along this flight, as a sample.

    Each is converted at *its own* station's altimeter setting, which is what
    lets two airports reporting different settings still be compared: pressure
    altitude is the coordinate they have in common.
    """
    samples = []
    for waypoint in waypoints:
        # Reported temperatures only: a fallback here would hand the profile
        # its own default back as though somebody had observed it.
        air = field_weather(waypoint, conditions, require_reported_temperature=True)
        if air is None:
            continue
        samples.append(
            TemperatureSample(air.pressure_altitude_ft, air.isa_deviation_c)
        )
    return samples


def _row_temperature_samples(
    segments: list[ProfileSegment],
    row_temperatures: dict[int, float],
    row_offset: int,
    conditions: Conditions,
) -> list[TemperatureSample]:
    """Hang each row's typed temperature at the altitude that row is flown at.

    Walks the draft profile with the same short-leg skip the real walk uses,
    so the row a temperature was typed against is the row it is read off.
    """
    samples = []
    row = row_offset
    for segment in segments:
        if inverse(segment.start.position, segment.end.position).distance_nm < _MIN_ROW_NM:
            continue
        oat_c = row_temperatures.get(row)
        if oat_c is not None:
            samples.append(
                TemperatureSample.observed(
                    segment.altitude_ft, oat_c, conditions.altimeter_inhg
                )
            )
        row += 1
    return samples


def _row_count(segments: list[ProfileSegment]) -> int:
    """How many of these segments are long enough to earn a navlog row."""
    return sum(
        1
        for s in segments
        if inverse(s.start.position, s.end.position).distance_nm >= _MIN_ROW_NM
    )


def _isa_deviation_at(conditions: Conditions, indicated_altitude_ft: float) -> float:
    """How far off standard the air is at one altitude.

    Referenced to *pressure* altitude, matching both `Conditions.oat_c` and the
    POH cruise interpolator, so that carrying a deviation from one altitude to
    another keeps the same airmass rather than quietly shifting it.
    """
    pa = conditions.pressure_altitude_ft(indicated_altitude_ft)
    return conditions.oat_c(indicated_altitude_ft) - isa_temperature_c(pa)


def _cruise_point_at(
    indicated_altitude_ft: float,
    *,
    aircraft: Aircraft,
    conditions: Conditions,
    default: perf.CruisePoint,
    pressure_offset_ft: float = 0.0,
) -> perf.CruisePoint:
    """The cruise chart at one altitude, or the flight's entry if unreadable.

    Clamped into the chart rather than refused: the callers are the pattern,
    the reserve and the top of a descent, none of which is worth failing a
    whole plan over, and the first two of which happen low -- usually below the
    chart's 2000 ft floor. Reading the lowest published page there is the
    conservative direction anyway, since the engine makes more power low down
    and so burns more.

    `pressure_offset_ft` carries a pressure altitude the pilot typed on the row
    this is being read for, the same way `_worst_level_gph` does: what
    transfers to another altitude is how far the row's air sits from the
    route-wide altimeter setting, not the number itself.
    """
    floor, ceiling = perf.cruise_altitude_range()
    pressure_alt = min(
        max(
            conditions.pressure_altitude_ft(indicated_altitude_ft)
            + pressure_offset_ft,
            floor,
        ),
        ceiling,
    )
    isa_dev = _isa_deviation_at(conditions, indicated_altitude_ft)
    try:
        # Read at an equal density where the day is off the chart's temperature
        # band, which down in the pattern on a hot afternoon it often is. A
        # reading from the published grid beats the fallback below, and the
        # fallback is still there for when even that has nothing to offer.
        return perf.cruise_at_density(
            pressure_alt,
            aircraft.cruise_rpm,
            isa_temperature_c(pressure_alt) + isa_dev,
        ).point
    except perf.OutsidePOHEnvelope:
        # The RPM is not published this low, or the day is off the chart. The
        # flight's own setting is a better answer than no plan at all.
        return default


def _reserve_gph(legs: list[Leg], default_gph: float) -> float:
    """The fuel flow the FAR 91.151 reserve is measured at.

    The regulation says "at normal cruising speed", so it is the flight's own
    cruise that sets it -- not an altitude typed into a box that the aeroplane
    may never reach. Where a plan cruises at more than one altitude the
    thirstiest is used, because the reserve is the fuel that has to still be
    there at the end and there is no credit for having planned an economical
    leg earlier.

    Falls back to the flight's nominal setting for a profile with no level
    flight in it at all -- a short hop that climbs and descends and nothing
    else, which has no cruise to measure.
    """
    rates = [
        leg.fuel_gal / (leg.ete_min / 60.0)
        for leg in legs
        if leg.phase == "cruise" and leg.ete_min > 0
    ]
    return max(rates) if rates else default_gph


def _altitude_overrides_by_leg(
    route: list[Waypoint], overrides: dict[int, LegOverride], row_offset: int
) -> dict[int, float]:
    """Re-key altitude overrides from navlog row to leg index.

    The two differ only where a leg is too short to earn a row, so the same
    skip rule the walk uses has to be applied here -- otherwise an override
    typed against one row would land on a different leg.
    """
    by_leg: dict[int, float] = {}
    row = row_offset
    for index, (start, end) in enumerate(pairwise(route)):
        if inverse(start.position, end.position).distance_nm < _MIN_ROW_NM:
            continue
        override = overrides.get(row)
        if override is not None and override.altitude_ft is not None:
            by_leg[index] = override.altitude_ft
        row += 1
    return by_leg


def _row_winds(overrides: dict[int, LegOverride]) -> dict[int, tuple]:
    """Every row that carries a wind, as the halves the pilot actually typed.

    Halves rather than a finished `TypedWind`, because what an untyped half
    falls back to depends on the leg: with a forecast column over that leg it
    is that column, and only without one is it the route's own profile. The
    base is therefore chosen where the leg is known, not here.
    """
    return {
        row: (o.wind_from_deg, o.wind_speed_kt)
        for row, o in overrides.items()
        if o.wind_from_deg is not None or o.wind_speed_kt is not None
    }


def _drawn_leg_of(spans: tuple[Segment, ...], point: LatLon) -> int | None:
    """Which leg the pilot drew a point sits on, or `None` if it sits on none.

    Containment, not nearest: a point belongs to the stretch of route it is
    *on*. Measuring to the nearest column instead hands the first ten miles of
    a 130 nm leg to the leg before it, because that leg's column -- sitting at
    its own midpoint, sixty miles back -- really is the closer of the two. The
    row is still unambiguously on the second leg.

    Where two legs both contain the point, which happens at the waypoint
    joining them, the one it deviates from least wins.
    """
    on = [
        (abs(span.cross_track_nm(point)), index)
        for index, span in enumerate(spans)
        if -_MIN_ROW_NM <= span.along_track_nm(point) <= span.distance_nm + _MIN_ROW_NM
    ]
    return min(on)[1] if on else None


@dataclass(frozen=True)
class _RouteColumns:
    """One flight's forecast columns, filed under the legs the pilot drew.

    Both halves of the filing are done by position: a column is put on the leg
    it was forecast over, and a row is given the column of the leg it flies
    along. Nothing is keyed by row or by index into the resolved route, which
    is what lets the forecast move the tops of climb -- and it does -- without
    moving out from under itself.
    """

    spans: tuple[Segment, ...]
    by_leg: dict[int, WindsAloft]

    @classmethod
    def build(
        cls, waypoints: list[Waypoint], field: WindField | None
    ) -> _RouteColumns | None:
        """`None` where there is no forecast, or none of it is on this flight.

        A route with a stop is planned as several flights, and each gets only
        the columns over its own legs.
        """
        if field is None or not field.columns:
            return None
        spans = tuple(
            inverse(start.position, end.position)
            for start, end in pairwise(strip_generated(waypoints))
        )
        by_leg: dict[int, WindsAloft] = {}
        for column in field.columns:
            index = _drawn_leg_of(spans, column.position)
            if index is not None:
                by_leg[index] = column.winds
        return cls(spans, by_leg) if by_leg else None

    def over(self, start: Waypoint, end: Waypoint) -> WindsAloft | None:
        """The column over a stretch of route, found by where that stretch is."""
        midpoint = inverse(start.position, end.position).point_at_fraction(0.5)
        index = _drawn_leg_of(self.spans, midpoint)
        return None if index is None else self.by_leg.get(index)

    def all_phases(self, winds: WindsAloft) -> dict[str, object]:
        """One column, offered to every phase the leg might be flown in.

        A resolved leg has exactly one phase and a drawn leg can hold three,
        and the lookup is by phase either way -- so the column answers to all
        of them and the leg takes the one it needs.
        """
        return dict.fromkeys(CONCRETE_SEGMENT_TYPES, winds)


def _leg_winds_by_leg(
    route: list[Waypoint],
    row_winds: dict[int, tuple],
    row_offset: int,
    conditions: Conditions,
    columns: _RouteColumns | None,
) -> dict[int, dict[str, object]]:
    """The wind each leg of a resolved route is flown in.

    Two things arrive here. The forecast column over the leg, which belongs to
    it by position and needs no keying; and a wind the pilot typed against a
    row, which does. The same skip rule `_altitude_overrides_by_leg` applies
    to the second, and for the same reason: a leg too short to earn a row must
    not shift a typed wind onto its neighbour.

    Typed beats forecast on the phase it was typed against, and a half-typed
    wind reads its other half off that leg's own column. Keyed by phase as
    well as leg because that is the shape `resolve_route` needs.
    """
    by_leg: dict[int, dict[str, object]] = {}
    row = row_offset
    for index, (start, end) in enumerate(pairwise(route)):
        if inverse(start.position, end.position).distance_nm < _MIN_ROW_NM:
            continue
        column = None if columns is None else columns.over(start, end)
        entry: dict[str, object] = (
            columns.all_phases(column) if column is not None else {}
        )
        typed = row_winds.get(row)
        if typed is not None:
            entry[end.segment_type] = TypedWind(column or conditions.winds, *typed)
        if entry:
            by_leg[index] = entry
        row += 1
    return by_leg


def _forecast_winds_by_drawn_leg(
    columns: _RouteColumns | None,
) -> dict[int, dict[str, object]]:
    """The forecast column over each leg the *pilot* drew.

    Needs no lay-out at all, unlike a typed wind: a column belongs to a leg by
    where it was forecast, and the legs the pilot drew are known before
    anything has been planned. So the planner places the tops of climb in the
    forecast wind from the first pass rather than the second.
    """
    if columns is None:
        return {}
    return {
        index: columns.all_phases(winds) for index, winds in columns.by_leg.items()
    }


def _leg_winds_by_drawn_leg(
    waypoints: list[Waypoint],
    draft_segments: list[ProfileSegment],
    row_winds: dict[int, tuple],
    row_offset: int,
    conditions: Conditions,
    columns: _RouteColumns | None,
) -> dict[int, dict[str, object]]:
    """Re-key row winds onto the legs the *pilot* drew, for the planner.

    Automatic planning decides where a top of climb falls before any of its
    rows exist, walking the pilot's own waypoints -- so a wind typed against a
    row has to be found a home on the leg that row sits inside. The draft
    lay-out is what says which that is: each of its segments is placed by
    ground position, and the leg containing its midpoint is the leg the pilot
    drew it on.

    Keyed by phase within the leg, because one drawn leg can hold both a climb
    and the descent off it, each with its own wind typed against its own row.
    The forecast for that leg goes underneath, so a typed direction with no
    speed takes its speed from the column over that leg.
    """
    drawn = list(pairwise(strip_generated(waypoints)))
    spans = [inverse(start.position, end.position) for start, end in drawn]

    by_leg = _forecast_winds_by_drawn_leg(columns)
    row = row_offset
    for segment in draft_segments:
        geo = inverse(segment.start.position, segment.end.position)
        if geo.distance_nm < _MIN_ROW_NM:
            continue
        typed = row_winds.get(row)
        row += 1
        if typed is None:
            continue
        midpoint = geo.point_at_fraction(0.5)
        for index, span in enumerate(spans):
            along = span.along_track_nm(midpoint)
            if -_MIN_ROW_NM <= along <= span.distance_nm + _MIN_ROW_NM:
                column = None if columns is None else columns.by_leg.get(index)
                by_leg.setdefault(index, {})[segment.phase] = TypedWind(
                    column or conditions.winds, *typed
                )
                break
    return by_leg


def _finish_navlog(
    *,
    legs: list[Leg],
    overrides: dict[int, LegOverride],
    aircraft: Aircraft,
    conditions: Conditions,
    cruise_point: perf.CruisePoint,
    cruise_altitude_ft: float,
    planning_mode: str,
    resolved_waypoints: list[Waypoint],
    travelled: float,
    cumulative_time: float,
    cumulative_fuel: float,
    warnings: list[str],
    checklist: preflight.GoNoGo,
) -> Navlog:
    """Totals, the reserve check, and the notes about manual edits."""
    # --- manual overrides -------------------------------------------------
    if overrides:
        edited = sum(1 for leg in legs if leg.overridden)
        if edited:
            warnings.append(
                f"{edited} of {len(legs)} rows use manually entered wind or true "
                f"airspeed rather than the computed values"
            )
        # An index past the end is a stale edit -- the route shrank under it.
        # Silently ignoring it would show the pilot an unedited row they
        # believe they edited, so say so.
        stale = sorted(i for i in overrides if not 0 <= i < len(legs))
        if stale:
            warnings.append(
                f"manual edits for row(s) {', '.join(str(i + 1) for i in stale)} "
                f"were discarded: the route no longer has those rows"
            )

    # --- reserve check ---------------------------------------------------
    # Measured at the flight's own cruise, not at the altitude typed into the
    # box -- see `_reserve_gph`.
    reserve_gal = _fuel_written_on_the_log(
        _reserve_gph(legs, cruise_point.gph) * conditions.reserve_minutes / 60.0
    )
    remaining = aircraft.fuel_on_board_gal - cumulative_fuel
    if remaining < reserve_gal:
        warnings.append(
            f"lands with {remaining:.1f} gal, below the "
            f"{conditions.reserve_minutes:.0f}-minute VFR reserve of "
            f"{reserve_gal:.1f} gal (FAR 91.151)"
        )
    if remaining < 0:
        warnings.append("the airplane runs out of fuel before the destination")

    return Navlog(
        legs=tuple(legs),
        total_distance_nm=travelled,
        total_time_min=cumulative_time,
        total_fuel_gal=cumulative_fuel,
        fuel_remaining_gal=remaining,
        reserve_required_gal=reserve_gal,
        cruise_altitude_ft=cruise_altitude_ft,
        planning_mode=planning_mode,
        resolved_waypoints=tuple(resolved_waypoints),
        warnings=tuple(warnings),
        checklist=checklist,
        conditions=conditions,
    )


def _leg_pressure_altitude(
    override: LegOverride | None,
    altitude_ft: float,
    conditions: Conditions,
    *,
    phase: str,
    leg_name: str,
    warnings: list[str],
) -> tuple[float, float]:
    """The pressure altitude one row's air is read at, and its offset.

    Normally the route's altimeter setting decides it. A pilot who types one
    instead is saying the air over this leg does not match that setting --
    which is an ordinary thing to find on a long route, and the whole reason
    the column is editable.

    The offset is returned alongside because a row can read the chart at more
    than one altitude: a climb that tops out mid-leg costs its level remainder
    at the *exit* altitude. Shifting that by the same amount keeps the row in
    one air mass, where using the typed number verbatim would put the level
    stretch at the climb's midpoint pressure.

    A climb is the one row this cannot re-cost: its time and fuel come from the
    POH climb table integrated over the whole climb, from the profile's
    altitudes, before any row exists. The entry still gives an honest density
    altitude there, and says so rather than letting a pilot believe otherwise.
    A level row reads its chart here, and a descent reads its fuel flow at the
    altitude it begins from, offset by this; both follow the number typed.
    """
    computed = conditions.pressure_altitude_ft(altitude_ft)
    if override is None or override.pressure_altitude_ft is None:
        return computed, 0.0
    typed = override.pressure_altitude_ft
    if phase == "climb":
        warnings.append(
            f"leg {leg_name} is a climb, so the pressure altitude of "
            f"{typed:.0f} ft entered against it sets its density altitude but "
            f"not its time and fuel, which come from the POH climb table read "
            f"over the whole climb."
        )
    return typed, typed - computed


def _leg_cruise_point(
    override: LegOverride | None,
    *,
    phase: str,
    pressure_altitude_ft: float,
    oat_c: float,
    entry_altitude_ft: float,
    pressure_offset_ft: float,
    conditions: Conditions,
    aircraft: Aircraft,
    default: perf.CruisePoint,
    leg_name: str,
    warnings: list[str],
) -> perf.CruisePoint:
    """The cruise chart entry one row is flown at.

    A cross-country is flown at **one power setting**: the RPM is the aircraft's
    throughout, and a row only departs from it by carrying an explicit
    `cruise_rpm`. But the chart is read at **this row's own** pressure altitude
    and temperature, so a leg levelling at 3,500 ft gets 3,500 ft numbers rather
    than the planned cruise altitude's. One lever position, read where the
    aeroplane actually is.

    The air comes in rather than being read off `conditions` here, so that the
    numbers printed on the row are the numbers the chart was read at -- both
    when the pilot typed them and when the model worked them out.

    A climb is flown at full throttle off the POH climb tables and a descent at
    whatever holds the target rate, so neither has a cruise setting of its own
    to look up. A climb keeps the flight's entry, which for it is only ever a
    fallback rate for a level remainder. A descent is charged at the cruise
    rate for want of a published descent table, and that rate is read **at the
    altitude the descent begins from** -- its own air, not the cruise altitude
    the pilot typed in a box, which in user-driven mode is only a bound and may
    be an altitude the aeroplane never sees.

    Note which way that errs: the top of a descent is its *thinnest* air, so
    the engine makes least power there and the rate read is the leanest of the
    ones the descent passes through. Reading the bottom instead would be the
    conservative end. This is the altitude the descent is entered at, which is
    what the row is named for.

    A row's own `cruise_rpm` is accepted on two conditions, and failing either
    is an error rather than a value quietly ignored -- the pilot asked for a
    power setting and needs to know it did not take:

    * the row is level flight
    * its pressure altitude is at or above
      `MIN_CUSTOM_CRUISE_RPM_PRESSURE_ALT_FT`
    """
    rpm = override.cruise_rpm if override is not None else None

    if phase != "cruise":
        if rpm is not None:
            raise RouteError(
                f"leg {leg_name}: a cruise RPM of {rpm:g} was set on a {phase} "
                f"row. Power settings apply to level flight only."
            )
        if phase == "descent":
            return _cruise_point_at(
                entry_altitude_ft,
                aircraft=aircraft,
                conditions=conditions,
                default=default,
                pressure_offset_ft=pressure_offset_ft,
            )
        return default

    pressure_alt = pressure_altitude_ft
    if rpm is not None and pressure_alt < MIN_CUSTOM_CRUISE_RPM_PRESSURE_ALT_FT:
        raise RouteError(
            f"leg {leg_name}: a cruise RPM of {rpm:g} was set at a pressure "
            f"altitude of {pressure_alt:.0f} ft, below the "
            f"{MIN_CUSTOM_CRUISE_RPM_PRESSURE_ALT_FT:.0f} ft floor for a "
            f"per-leg power setting."
        )

    chosen = rpm if rpm is not None else aircraft.cruise_rpm

    # A level row can sit below the bottom of the cruise chart, which starts at
    # 2000 ft. Read the lowest published page rather than refusing: the row is
    # real and has to be costed. Down there the engine makes *more* power than
    # that page shows, so the fuel flow is the optimistic side -- hence a
    # warning rather than a silent substitution.
    floor = perf.cruise_altitude_range()[0]
    if pressure_alt < floor:
        warnings.append(
            f"leg {leg_name} is level at a pressure altitude of "
            f"{pressure_alt:.0f} ft, below the {floor:.0f} ft bottom of the "
            f"cruise chart. Read at {floor:.0f} ft, which under-reads the fuel "
            f"flow slightly."
        )
        pressure_alt = floor

    def read(setting: float) -> perf.CruisePoint:
        """The chart at this row, moved onto a page that publishes its air.

        A hot afternoon leaves the chart's ISA+20 column at any altitude a 172
        cruises at, and refusing the flight for it would be absurd. See
        `perf.cruise_at_density`: the reading is still taken from inside the
        published grid, at the same density altitude, and the pilot is told
        where it came from.
        """
        lookup = perf.cruise_at_density(pressure_alt, setting, oat_c)
        if lookup.substituted:
            warnings.append(
                f"leg {leg_name} is level at a pressure altitude of "
                f"{pressure_alt:.0f} ft and ISA"
                f"{oat_c - isa_temperature_c(pressure_alt):+.0f}, outside the "
                f"cruise chart's published temperature band. Read at "
                f"{lookup.pressure_altitude_ft:.0f} ft and ISA"
                f"{lookup.isa_deviation_c:+.0f} instead, which is the same "
                f"density altitude of {lookup.density_altitude_ft:.0f} ft."
            )
        return lookup.point

    try:
        return read(chosen)
    except perf.OutsidePOHEnvelope as exc:
        # The chart is ragged: which RPMs exist narrows with altitude. Say
        # which ones do exist rather than only that this one does not. Asked
        # of the page the reading would actually come off, which on a hot day
        # is not the page this row's pressure altitude names.
        try:
            usable = perf.available_cruise_rpm(
                pressure_alt, oat_c, density_substitution=True
            )
        except perf.OutsidePOHEnvelope:
            usable = []
        if rpm is None and usable:
            # The pilot chose the setting but the planner chose this altitude,
            # so fly the nearest published one and say so rather than refusing
            # a route that is otherwise fine.
            nearest = min(usable, key=lambda r: abs(r - chosen))
            warnings.append(
                f"leg {leg_name} is level at a pressure altitude of "
                f"{pressure_alt:.0f} ft, where the POH does not publish "
                f"{chosen:g} RPM. Planned at {nearest:g} RPM instead."
            )
            return read(nearest)
        offer = (
            f" Published settings there: {', '.join(f'{r:g}' for r in usable)}."
            if usable
            else ""
        )
        raise RouteError(f"leg {leg_name}: {exc}.{offer}") from exc


def _apply_override(
    override: LegOverride | None,
    wind: Wind,
    tas_kt: float,
    leg_name: str,
) -> tuple[Wind, float, tuple[str, ...]]:
    """Fold a row's manual values over the computed ones.

    Returns the wind and true airspeed to use, plus the names of the fields
    that were replaced, so the row can say which of its numbers are the
    pilot's rather than the model's.
    """
    if override is None:
        return wind, tas_kt, ()

    overridden: list[str] = []
    from_deg, speed_kt = wind.from_deg, wind.speed_kt

    # The RPM's own effect is applied by `_leg_cruise_point` before this, and
    # the altitude's by `build_segments`; both reach here already folded into
    # `tas_kt`. They are marked so the row can say which numbers are the
    # pilot's.
    if override.cruise_rpm is not None:
        overridden.append("cruise_rpm")
    if override.altitude_ft is not None:
        overridden.append("altitude_ft")
    # The temperature reached the row the long way round -- through the flight's
    # temperature profile and a second lay-out -- so like the two above it is
    # already folded into `tas_kt` by the time this runs, and only needs naming.
    if override.oat_c is not None:
        overridden.append("oat_c")
    # Applied by `_leg_pressure_altitude` before this, and from there into the
    # charts the row reads and its own density altitude.
    if override.pressure_altitude_ft is not None:
        overridden.append("pressure_altitude_ft")

    if override.wind_from_deg is not None:
        from_deg = override.wind_from_deg % 360.0
        overridden.append("wind_from_deg")
    if override.wind_speed_kt is not None:
        if override.wind_speed_kt < 0:
            raise RouteError(
                f"leg {leg_name}: wind speed {override.wind_speed_kt:g} kt is negative"
            )
        speed_kt = override.wind_speed_kt
        overridden.append("wind_speed_kt")
    if override.tas_kt is not None:
        if override.tas_kt <= 0:
            raise RouteError(
                f"leg {leg_name}: true airspeed {override.tas_kt:g} kt must be positive"
            )
        tas_kt = override.tas_kt
        overridden.append("tas_kt")

    return Wind(from_deg, speed_kt), tas_kt, tuple(overridden)


def _fuel_written_on_the_log(gallons: float) -> float:
    """Fuel rounded **up** to the tenth of a gallon a pilot writes in the box.

    Every fuel figure on a paper navigation log is rounded up to the nearest
    tenth and the column is then added from those figures, so that is what this
    log does too: each row is rounded as it is emitted and the running total,
    the landing figure and the reserve check are all built from the rounded
    rows. Round for display alone and the column stops adding up, which on a
    log a pilot totals by hand is worse than being a tenth out.

    Up, never to nearest: a plan should not come out of the arithmetic holding
    less fuel than it started with. The tolerance keeps a figure that is
    already an exact tenth from being pushed to the next one by the last bit of
    a float -- 2.6 must not become 2.7.
    """
    return math.ceil(gallons * 10.0 - 1e-9) / 10.0


def _leg_fuel(
    phase: str,
    ete_min: float,
    climb: perf.ClimbSegment | None,
    cruise_gph: float,
    *,
    level_minutes: float = 0.0,
    level_gph: float | None = None,
) -> float:
    """Fuel for one leg, at the rate appropriate to its phase.

    A climb is charged the **published figure in full**, never scaled. The old
    code pro-rated it by `ete_min / climb.time_min`, which was written for the
    automatic planner -- where a climb segment *is* the POH climb, possibly
    split over two legs, so the ratio is at most one. In user-driven mode a leg
    is as long as the waypoints make it, the ratio can be ten, and the same
    formula charged ten times the book fuel. Worse, the ratio shrank as the
    climb grew, so fuel *fell* as the target altitude rose.

    Where the climb tops out before the waypoint the rest of the leg is level,
    and is charged at `level_gph` -- the most power the POH publishes at that
    altitude, not the economy setting the pilot planned. That stretch was not
    part of anybody's cruise plan, so the worst published case is the honest
    thing to assume for it.

    Descent is charged at the cruise rate: the POH publishes no descent table
    at all (see data/poh/c172s/SOURCE.md), and cruise is the conservative
    assumption -- a real 172 descending at partial power burns less. That rate
    is read at the altitude the descent begins from; see `_leg_cruise_point`.
    """
    if phase == "climb":
        if climb is None or climb.time_min <= 0:
            return 0.0
        if level_minutes <= 0.0:
            # The leg is the climb, or shorter than it. Charging a fraction is
            # right here: the rest of this climb is on the next leg.
            return climb.fuel_gal * min(1.0, ete_min / climb.time_min)
        return (
            climb.fuel_gal
            + (level_gph if level_gph is not None else cruise_gph)
            * level_minutes
            / 60.0
        )
    return cruise_gph * ete_min / 60.0


def _worst_level_gph(
    altitude_ft: float,
    *,
    conditions: Conditions,
    default: float,
    pressure_offset_ft: float = 0.0,
) -> float:
    """The thirstiest level fuel flow the POH admits to at this altitude.

    Clamped into the chart and falling back to the flight's own rate, for the
    same reason `_cruise_point_at` does: this costs a stretch of a leg, and is
    not worth failing a whole plan over.

    `pressure_offset_ft` carries a pressure altitude the pilot typed on this
    row: the level stretch is higher up than the row's own altitude, so what
    transfers is how far the row's air sits from the route-wide altimeter
    setting, not the number itself.
    """
    floor, ceiling = perf.cruise_altitude_range()
    pressure_alt = min(
        max(conditions.pressure_altitude_ft(altitude_ft) + pressure_offset_ft, floor),
        ceiling,
    )
    isa_dev = _isa_deviation_at(conditions, altitude_ft)
    try:
        return perf.max_published_cruise(
            pressure_alt, isa_temperature_c(pressure_alt) + isa_dev
        ).gph
    except perf.OutsidePOHEnvelope:
        return default


def leg_label(name: str, role: str | None) -> str:
    """A waypoint name with its top-of-climb/descent role, if it has one.

    `TOC` on its own where the planner invented the point, `KWVI (TOC)` where
    the pilot nominated a charted one -- the name on the sectional is what they
    will look for, so it is never replaced.
    """
    if role is None or name == role:
        return name
    return f"{name} ({role})"


def format_navlog(navlog: Navlog) -> str:
    """Render a navlog as fixed-width text, in the usual paper layout."""
    header = (
        f"{'FROM':<10}{'TO':<10}{'PHASE':<8}{'ALT':>6}{'TC':>6}{'WCA':>6}"
        f"{'TH':>6}{'VAR':>6}{'MH':>6}{'WIND':>10}{'TAS':>6}{'GS':>6}"
        f"{'DIST':>7}{'ETE':>6}{'FUEL':>6}{'REM':>6}"
    )
    lines = [header, "-" * 110]
    for leg in navlog.legs:
        if not leg.covers_ground:
            # Taxi and the pattern go nowhere: printing a course and a heading
            # for them would invite someone to fly one.
            lines.append(
                f"{leg.from_name:<10}{'':<10}{leg.phase:<8}"
                f"{leg.altitude_ft:>6.0f}{'--':>6}{'--':>6}{'--':>6}"
                f"{'--':>6}{'--':>6}{'--':>10}{'--':>6}{'--':>6}"
                f"{'--':>7}{leg.ete_min:>6.1f}"
                f"{leg.fuel_gal:>6.1f}{leg.fuel_remaining_gal:>6.1f}"
            )
            continue
        # A star marks a row whose wind or TAS the pilot entered by hand, so
        # the printed log cannot pass an edited row off as a computed one.
        wind = f"{leg.wind_from_deg:03.0f}/{leg.wind_speed_kt:02.0f}"
        if leg.overridden:
            wind = f"*{wind}"
        from_label = leg_label(leg.from_name, leg.start_role)
        to_label = leg_label(leg.to_name, leg.end_role)
        lines.append(
            f"{from_label:<10}{to_label:<10}{leg.phase:<8}"
            f"{leg.altitude_ft:>6.0f}{leg.true_course_deg:>6.0f}"
            f"{leg.wind_correction_angle_deg:>+6.1f}{leg.true_heading_deg:>6.0f}"
            f"{leg.variation_deg:>+6.1f}{leg.magnetic_heading_deg:>6.0f}"
            f"{wind:>10}{leg.tas_kt:>6.0f}{leg.ground_speed_kt:>6.0f}"
            f"{leg.distance_nm:>7.1f}{leg.ete_min:>6.1f}"
            f"{leg.fuel_gal:>6.1f}{leg.fuel_remaining_gal:>6.1f}"
        )
    lines.append("-" * 110)
    lines.append(
        f"{'TOTAL':<28}{'':>46}{navlog.total_distance_nm:>19.1f}"
        f"{navlog.total_time_min:>6.1f}{navlog.total_fuel_gal:>6.1f}"
        f"{navlog.fuel_remaining_gal:>6.1f}"
    )
    if navlog.warnings:
        lines.append("")
        lines.extend(f"WARNING: {w}" for w in navlog.warnings)
    if navlog.checklist is not None:
        lines.append("")
        lines.append(preflight.format_checklist(navlog.checklist))
    lines.append("")
    lines.append("NOT FOR NAVIGATION -- verify against the POH and current charts.")
    return "\n".join(lines)
