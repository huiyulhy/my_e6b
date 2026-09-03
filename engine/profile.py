"""The vertical profile: what each leg of a route does about altitude.

A route is a list of waypoints. This module turns it into a list of
`ProfileSegment` -- **exactly one per leg** -- each of which is a climb, a
cruise or a descent at a single altitude, airspeed and fuel flow. 

Two ways to get there, sharing one model:

* **User-driven.** The pilot declares each leg's `segment_type`. Where no
  altitude is given, it calculates the maximum altitude reachable based on POH
  ,same for descend at the configured rate.
* **Automatic.** Legs are declared `automatic`; `resolve_route` expands them
  against a target cruise altitude, **inserting TOC and TOD waypoints into the
  route** so that afterwards every leg is concrete and the user-driven walk
  runs unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from engine import performance as perf
from engine import preflight
from engine.atmosphere import tas_from_cas
from engine.geo import (
    LatLon,
    Segment,
    WindTooStrong,
    inverse,
    solve_wind_triangle,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from engine.navlog import Aircraft, Conditions, Wind

# Type of flight phase
SegmentType = Literal["climb", "cruise", "descent", "automatic"]

# A wind the pilot typed against one leg, keyed by leg index and then by the
# phase it was typed on. 
LegWinds = dict[int, dict[str, "object"]]

CONCRETE_SEGMENT_TYPES: tuple[str, ...] = ("climb", "cruise", "descent")
PLANNING_MODES: tuple[str, ...] = ("manual", "auto")

# Below this, an altitude change is not worth a row of its own.
_ALTITUDE_EPSILON_FT = 1.0
# A boundary this close to the end of a leg is the end of the leg.
_BOUNDARY_EPSILON_NM = 0.1
# The step an altitude change is integrated on. The POH climb tables are
# printed every 1000 ft, and marching on that spacing lands within 0.03 nm of a
# 100 ft march even in strong shear
_PROFILE_BAND_FT = 1000.0


class RouteError(ValueError):
    """The route cannot be flown as given."""


@dataclass(frozen=True)
class Waypoint:
    """A point on the route.

    `kind` drives both the navlog's checkpoint column and, later, the router's
    landmark preference -- a published VFR waypoint is far easier to identify
    from the air than an arbitrary fix.
    """

    name: str
    position: LatLon
    kind: str = "waypoint"  # airport | vor | ndb | fix | vfr_waypoint | town | phase
    elevation_ft: float | None = None
    frequency: str | None = None
    notes: str = ""

    # Set to True if waypoint is a stopover airport and we plan to land
    # Flag is ignored on the first and last waypoint (departure and landing) destination
    is_landing: bool = False

    # Every runway on the field, for the go/no-go check. Empty means unknown,
    # which the checklist reports as unknown rather than treating as a pass.
    runways: tuple[preflight.Runway, ...] = ()

    # Cross this waypoint at this altitude. 
    altitude_ft: float | None = None

    # What the leg arriving at this waypoint does about altitude. Ignored on
    # the first waypoint of a flight, which is departed from rather than
    # arrived at
    segment_type: str = "automatic"

    # True for a TOC/TOD the automatic planner inserted. `resolve_route`
    # discards these before re-expanding
    generated: bool = False

    # Field weather, for airports. The route's altimeter setting and ISA
    # deviation describe the air en route; used to calculate density altitude.
    altimeter_inhg: float | None = None
    oat_c: float | None = None

    # The surface wind on the field, TRUE-referenced as a METAR or TAF reports
    # it. The go/no-go check converts it to magnetic to line it up with the
    # runway designators. Null means unknown, and the runway distances fall
    # back to the no-wind book figures, which the checklist says out loud.
    wind_from_deg: float | None = None
    wind_speed_kt: float | None = None
    gust_kt: float | None = None

    # The rest of the field's report, for the go/no-go's VFR check. Only the
    # parts a VFR decision turns on: how far you can see, how low the cloud
    # is, and what kind of cloud it is -- an overcast has to clear the pattern
    # and a vertical visibility is not something to fly under at all.
    #
    # `sky_reported` is separate from the ceiling on purpose. A clear sky and
    # a source that does not observe cloud both arrive with no ceiling, and
    # the checklist has to call the first VFR and the second unknown.
    visibility_sm: float | None = None
    ceiling_ft_agl: float | None = None
    ceiling_cover: str = ""  # BKN | OVC | OVX | VV; "" when there is no ceiling
    sky_reported: bool = False

    # Whether a report was obtained for this field at all. The distinction the
    # empty values cannot carry: a model-only forecast answers with no cloud
    # and no visibility in it, which looks exactly like never having asked.
    # The first is an unknown the pilot should see; the second is silence.
    weather_reported: bool = False

    # The field's published traffic pattern altitude in feet AGL, which sets
    # how high an overcast has to be. `None` takes the standard 1,000 ft.
    pattern_altitude_agl_ft: float | None = None

    @property
    def field_weather_report(self) -> preflight.FieldWeather | None:
        """The report as the go/no-go reads it, or `None` if there is none.

        None only where nothing was ever asked for. A report that arrived
        saying nothing useful is still a report, and it comes back as an
        unknown rather than as an absence -- a model forecast observes no
        cloud, and "no cloud observed" is not "no cloud".
        """
        if not (
            self.weather_reported
            or self.visibility_sm is not None
            or self.ceiling_ft_agl is not None
            or self.ceiling_cover
            or self.sky_reported
        ):
            return None
        return preflight.FieldWeather(
            visibility_sm=self.visibility_sm,
            ceiling_ft_agl=self.ceiling_ft_agl,
            ceiling_cover=self.ceiling_cover,
            sky_reported=self.sky_reported,
        )


@dataclass(frozen=True)
class ProfileSegment:
    """One row's worth of flying: one leg, at one phase.

    After resolution there is exactly one of these per leg, which is what
    makes navlog rows correspond 1:1 to legs
    """

    start: Waypoint
    end: Waypoint
    phase: str  # climb | cruise | descent
    altitude_ft: float  # representative altitude: Used for calculating density
    # altitude and interpolating later
    entry_altitude_ft: float
    exit_altitude_ft: float
    tas_kt: float
    # The POH climb segment this piece came from, so its fuel can be charged
    # from the published figure. `None` for level and descending flight.
    climb: perf.ClimbSegment | None
    # "TOC" / "TOD" when this segment's boundary is a top of climb or descent.
    # Carried rather than baked into the waypoint name, so a charted point used
    # as a top of climb keeps the name printed on the sectional.
    start_role: str | None = None
    end_role: str | None = None

    # Ground distance flown *level* on this leg, after the climb has topped
    # out. Non-zero only in user-driven mode
    # Kept as a distance because the two cost different fuel: 
    # the climb is charged the published figure, and this part is charged
    #  as level flight.
    level_distance_nm: float = 0.0
    level_minutes: float = 0.0


class PhaseNamer:
    """Names the inserted points: TOC, TOD, TOC2, TOD2 ...
    """

    def __init__(self) -> None:
        self._counts = {"climb": 0, "descent": 0}

    def next(self, phase: str) -> str:
        self._counts[phase] += 1
        n = self._counts[phase]
        stem = "TOC" if phase == "climb" else "TOD"
        return stem if n == 1 else f"{stem}{n}"


# --- resolution: every leg gets a concrete type --------------------------
def strip_generated(waypoints: list[Waypoint]) -> list[Waypoint]:
    """Drop planner-inserted points, leaving the pilot's own route.

    Resolution starts here every time. Without it, re-planning an already
    expanded route would insert a second TOC beside the first and the edit
    loop would never settle.
    """
    return [w for w in waypoints if not w.generated]


def resolve_route(
    waypoints: list[Waypoint],
    *,
    mode: str,
    cruise_altitude_ft: float,
    departure_elevation: float,
    destination_elevation: float,
    aircraft: Aircraft,
    conditions: Conditions,
    names: PhaseNamer,
    warnings: list[str],
    leg_winds: LegWinds | None = None,
) -> list[Waypoint]:
    """Return a route on which every leg has a concrete segment type.

    `leg_winds` is keyed by leg of *this* route -- the one the pilot drew --
    because that is what the planner walks. `build_segments` takes the same
    map keyed by the resolved route it produces, which is not the same list
    once tops of climb have been inserted into it.
    """
    if mode not in PLANNING_MODES:
        raise RouteError(f"unknown planning mode {mode!r}")

    route = strip_generated(waypoints)
    if len(route) < 2:
        raise RouteError("a route needs at least a departure and a destination")

    if mode == "manual":
        undeclared = [
            w.name for w in route[1:] if w.segment_type not in CONCRETE_SEGMENT_TYPES
        ]
        if undeclared:
            plural = "is" if len(undeclared) == 1 else "are"
            raise RouteError(
                "user-driven planning needs a segment type on every leg; "
                f"{', '.join(undeclared)} {plural} still automatic. "
                f"Choose climb, cruise or descent."
            )
        return route

    return _expand_automatic(
        route,
        leg_winds=leg_winds,
        cruise_altitude_ft=cruise_altitude_ft,
        departure_elevation=departure_elevation,
        destination_elevation=destination_elevation,
        aircraft=aircraft,
        conditions=conditions,
        names=names,
        warnings=warnings,
    )

def _expand_automatic(
    waypoints: list[Waypoint],
    *,
    leg_winds: LegWinds | None = None,
    cruise_altitude_ft: float,
    departure_elevation: float,
    destination_elevation: float,
    aircraft: Aircraft,
    conditions: Conditions,
    names: PhaseNamer,
    warnings: list[str],
) -> list[Waypoint]:
    """Insert TOC/TOD points until every leg is a single concrete phase.
    Two passes, because a leg cannot be planned in isolation. Waypoints do not
    fall where the climb and descent happen to end, so a leg may not finish its
    climb, and a descent may have to begin several legs before the one it lands
    on.

    Backwards first: the highest altitude each waypoint may be crossed at and
    still get down. Then forwards, carrying the altitude actually reached
    rather than the one that was aimed for.
    """
    plateaus = _plateau_altitudes(waypoints, cruise_altitude_ft)
    ceilings = _descent_ceilings(
        waypoints,
        plateaus,
        destination_elevation=destination_elevation,
        aircraft=aircraft,
        conditions=conditions,
        leg_winds=leg_winds,
    )

    resolved: list[Waypoint] = [waypoints[0]]
    entry = departure_elevation
    for index, (start, end) in enumerate(pairwise(waypoints)):
        # Never aim higher than the point from which the rest of the route can
        # still be flown down.
        plateau = min(plateaus[index], ceilings[index])
        exit_ = ceilings[index + 1]

        inserted, end_type, entry = _expand_leg(
            start,
            end,
            entry_altitude_ft=entry,
            plateau_altitude_ft=plateau,
            exit_altitude_ft=exit_,
            aircraft=aircraft,
            conditions=conditions,
            winds_by_phase=(leg_winds or {}).get(index),
            names=names,
            warnings=warnings,
        )
        resolved.extend(inserted)
        # Only the *type* is written back onto the pilot's own waypoints, never
        # an altitude. Writing an altitude would make the next resolve read it
        # as a crossing restriction, which changes the plateau, which changes
        # where the TOD falls -- the route would drift on every re-plan instead
        # of settling. The inserted points carry the altitudes instead; they are
        # discarded and re-derived each time, so they cannot accumulate error.
        resolved.append(replace(end, segment_type=end_type))
    return resolved


def _expand_leg(
    start: Waypoint,
    end: Waypoint,
    *,
    entry_altitude_ft: float,
    plateau_altitude_ft: float,
    exit_altitude_ft: float,
    aircraft: Aircraft,
    conditions: Conditions,
    winds_by_phase: dict[str, object] | None = None,
    names: PhaseNamer,
    warnings: list[str],
) -> tuple[list[Waypoint], str, float]:
    """Split one leg, returning the points to insert before `end`.

    Also returns the phase of the piece that *arrives* at `end`, and the
    altitude actually reached there -- which is not always the one asked for.
    A waypoint dropped three miles into a fourteen mile climb leaves the
    aeroplane part way up, and the caller carries that forward so the climb
    continues on the next leg and the top of climb lands where it really falls.
    """
    geo = inverse(start.position, end.position)
    if geo.distance_nm <= 0:
        return [], "cruise", entry_altitude_ft

    # A leg the pilot drew can hold two changes -- up to the plateau and back
    # down off it -- and a wind was typed against each of them separately, so
    # each is solved in its own. Which is which comes from the direction the
    # piece goes, not from what the leg is called.
    lead_conditions = _in_phase_wind(
        conditions, winds_by_phase, entry_altitude_ft, plateau_altitude_ft
    )
    tail_conditions = _in_phase_wind(
        conditions, winds_by_phase, plateau_altitude_ft, exit_altitude_ft
    )
    lead = _solve_change(
        entry_altitude_ft, plateau_altitude_ft, geo, aircraft, lead_conditions
    )
    tail = _solve_change(
        plateau_altitude_ft, exit_altitude_ft, geo, aircraft, tail_conditions
    )

    if lead is None and tail is None:
        return [], "cruise", plateau_altitude_ft

    # The leg is still changing altitude when it ends: it runs out of distance
    # before the plateau, and there is no descent to fit in afterwards. Fly as
    # much of the change as fits and hand the rest to the next leg. This is the
    # ordinary case for a waypoint placed inside the climb, so it is not a
    # warning -- nothing is wrong with the route.
    if lead is not None and tail is None and lead["distance_nm"] > geo.distance_nm:
        achieved = reachable_altitude(
            entry_altitude_ft,
            plateau_altitude_ft,
            geo.distance_nm,
            geo,
            aircraft,
            lead_conditions,
        )
        piece = _solve_change(
            entry_altitude_ft, achieved, geo, aircraft, lead_conditions
        )
        if piece is None:
            return [], "cruise", entry_altitude_ft
        return [], piece["phase"], achieved

    needed = sum(p["distance_nm"] for p in (lead, tail) if p)
    if needed > geo.distance_nm:
        warnings.append(
            f"{start.name} to {end.name} is {geo.distance_nm:.0f} nm but the "
            f"altitude changes on it need {needed:.0f} nm; the airplane never "
            f"levels off. Choose a lower altitude or move the waypoint."
        )
        # Scale in proportion rather than letting the pieces overlap.
        scale = geo.distance_nm / needed
        for piece in (lead, tail):
            if piece:
                piece["distance_nm"] *= scale
        needed = geo.distance_nm

    # Lay the pieces out along the leg as (piece, distance-at-which-it-ends).
    # A climb out of the start comes first, a descent into the end comes last,
    # and whatever distance is left over in between is flown level.
    plan: list[tuple[dict | None, float]] = []
    cursor = 0.0
    if lead:
        cursor += lead["distance_nm"]
        plan.append((lead, cursor))
    level_distance = geo.distance_nm - needed
    if level_distance > _BOUNDARY_EPSILON_NM:
        cursor += level_distance
        plan.append((None, cursor))
    if tail:
        plan.append((tail, geo.distance_nm))

    inserted: list[Waypoint] = []
    for index, (piece, ends_at) in enumerate(plan[:-1]):
        # The point that ends this piece is named for the phase it hands over
        # to: the end of a climb is the top of climb, and the end of a level
        # stretch before a descent is the top of descent.
        handover = plan[index + 1][0]
        phase = piece["phase"] if piece else handover["phase"]
        inserted.append(
            Waypoint(
                name=names.next(phase),
                position=geo.point_at_nm(ends_at),
                kind="phase",
                # The altitude is carried on the point itself, so rebuilding
                # the profile from the resolved route reproduces it exactly.
                altitude_ft=plateau_altitude_ft,
                segment_type=piece["phase"] if piece else "cruise",
                generated=True,
            )
        )

    final_piece = plan[-1][0]
    end_type = final_piece["phase"] if final_piece else "cruise"
    achieved = exit_altitude_ft if tail else plateau_altitude_ft
    return inserted, end_type, achieved


# --- building the segments -----------------------------------------------


def _in_leg_wind(
    conditions: Conditions,
    leg_winds: LegWinds | None,
    index: int,
    phase: str,
) -> Conditions:
    """`conditions` as this leg is flown, with a typed wind swapped in.

    The wind a pilot types on a navlog row is a statement about that leg, not
    about an altitude: it holds all the way up the climb and all the way along
    the ground the row covers. So it replaces the route's wind profile outright
    for the leg it was typed on, rather than being hung at one altitude and
    interpolated between -- which is what `Conditions.temperatures` does, and is
    right for temperature and wrong for this.
    """
    winds = (leg_winds or {}).get(index, {}).get(phase)
    return conditions if winds is None else replace(conditions, winds=winds)


def _in_phase_wind(
    conditions: Conditions,
    winds_by_phase: dict[str, object] | None,
    from_altitude_ft: float,
    to_altitude_ft: float,
) -> Conditions:
    """`_in_leg_wind` for one piece of a leg, named by which way it goes."""
    if not winds_by_phase:
        return conditions
    phase = "climb" if to_altitude_ft > from_altitude_ft else "descent"
    winds = winds_by_phase.get(phase)
    return conditions if winds is None else replace(conditions, winds=winds)


def build_segments(
    waypoints: list[Waypoint],
    *,
    departure_elevation: float,
    aircraft: Aircraft,
    conditions: Conditions,
    cruise_point: perf.CruisePoint,
    altitude_overrides: dict[int, float] | None = None,
    leg_winds: LegWinds | None = None,
) -> list[ProfileSegment]:
    """One segment per leg, from a route whose legs all have concrete types.

    The exit altitude of each leg is the entry altitude of the next, so the
    profile is continuous by construction. Where a waypoint states an altitude
    that is what it gets, **even when it contradicts the declared type** -- a
    "climb to" that ends lower than it started is emitted as declared and left
    for `consistency.check_navlog_consistency` to report. Building and checking
    are kept apart on purpose: a planner that quietly corrects the pilot hides
    the mistake instead of showing it.

    `altitude_overrides` maps a leg index to an altitude the pilot typed into
    the finished log. It is applied here rather than afterwards so the change
    carries forward: editing the top of a climb re-flies every leg after it,
    which is the whole point of editing it.

    `leg_winds` does the same for a wind typed against a row. It matters most
    on a "climb" with no stated altitude, where the exit altitude *is* the top
    the aeroplane reaches in the distance available: into a headwind there are
    fewer ground miles per minute of climb, so the leg tops out lower.
    """
    altitude_overrides = altitude_overrides or {}
    segments: list[ProfileSegment] = []
    entry = departure_elevation

    for index, (start, end) in enumerate(pairwise(waypoints)):
        phase = end.segment_type
        if phase not in CONCRETE_SEGMENT_TYPES:
            raise RouteError(
                f"leg {start.name} to {end.name} is still {phase!r}; "
                f"resolve the route before building segments"
            )

        geo = inverse(start.position, end.position)
        # Every reading for this leg -- the top it reaches and the marching
        # that gets it there -- is taken in the leg's own wind.
        leg_conditions = _in_leg_wind(conditions, leg_winds, index, phase)
        exit_altitude = altitude_overrides.get(index)
        if exit_altitude is None:
            exit_altitude = _exit_altitude(
                end,
                phase,
                entry_altitude_ft=entry,
                geo=geo,
                aircraft=aircraft,
                conditions=leg_conditions,
            )
        segments.append(
            _segment_for(
                start,
                end,
                phase=phase,
                entry_altitude_ft=entry,
                exit_altitude_ft=exit_altitude,
                geo=geo,
                aircraft=aircraft,
                conditions=leg_conditions,
                cruise_point=cruise_point,
            )
        )
        entry = exit_altitude

    return _mark_boundaries(segments)


def _exit_altitude(
    end: Waypoint,
    phase: str,
    *,
    entry_altitude_ft: float,
    geo: Segment,
    aircraft: Aircraft,
    conditions: Conditions,
) -> float:
    """Where this leg leaves the aeroplane."""
    if end.altitude_ft is not None:
        return end.altitude_ft
    if phase == "cruise":
        return entry_altitude_ft
    if phase == "climb":
        return reachable_altitude(
            entry_altitude_ft,
            _climb_ceiling(conditions),
            geo.distance_nm,
            geo,
            aircraft,
            conditions,
        )
    return _descent_reachable(
        entry_altitude_ft,
        end.elevation_ft if end.elevation_ft is not None else 0.0,
        geo.distance_nm,
        geo,
        aircraft,
        conditions,
    )


def _climb_ceiling(conditions: Conditions) -> float:
    """The highest altitude the POH climb table will answer for.

    A "climb to" with no altitude means "as high as you get", so it needs a
    target to bisect toward. The chart's top row is the honest one -- above it
    the engine would have to extrapolate, which it refuses to do.
    """
    top = perf.climb_table_ceiling_ft()
    # The table is indexed by pressure altitude; the caller works in indicated.
    return top - (conditions.pressure_altitude_ft(0.0) - 0.0)


def _segment_for(
    start: Waypoint,
    end: Waypoint,
    *,
    phase: str,
    entry_altitude_ft: float,
    exit_altitude_ft: float,
    geo: Segment,
    aircraft: Aircraft,
    conditions: Conditions,
    cruise_point: perf.CruisePoint,
) -> ProfileSegment:
    """Airspeed and POH climb figures for one declared leg."""
    mid = 0.5 * (entry_altitude_ft + exit_altitude_ft)
    change = _solve_change(
        entry_altitude_ft, exit_altitude_ft, geo, aircraft, conditions
    )
    level_nm = level_minutes = 0.0

    if phase == "climb" and change is not None and change["phase"] == "climb":
        change = _fitted_to_the_leg(
            change, entry_altitude_ft, exit_altitude_ft, geo, aircraft, conditions
        )
        tas_kt, climb = change["tas_kt"], change["climb"]
        tas_kt, level_nm, level_minutes = _with_level_remainder(
            change, geo, conditions, cruise_point, exit_altitude_ft
        )
    elif phase == "descent" and change is not None and change["phase"] == "descent":
        tas_kt, climb = change["tas_kt"], None
    elif phase == "cruise":
        tas_kt, climb = cruise_point.ktas, None
    else:
        # A declared type the altitudes contradict -- a "climb" that descends,
        # or one with no altitude change at all. Fly it at the speed its
        # geometry implies and let the consistency check name the problem.
        tas_kt = change["tas_kt"] if change is not None else cruise_point.ktas
        climb = change["climb"] if change is not None else None

    return ProfileSegment(
        start=start,
        end=end,
        phase=phase,
        altitude_ft=mid,
        entry_altitude_ft=entry_altitude_ft,
        exit_altitude_ft=exit_altitude_ft,
        tas_kt=tas_kt,
        climb=climb,
        level_distance_nm=level_nm,
        level_minutes=level_minutes,
    )


def _fitted_to_the_leg(
    change: dict,
    entry_altitude_ft: float,
    exit_altitude_ft: float,
    geo: Segment,
    aircraft: Aircraft,
    conditions: Conditions,
) -> dict:
    """Charge only the climb the leg has room for.

    A declared climb can ask for more height than the distance allows -- "climb
    to 5000" on a four mile leg. The declared altitude is left standing, so
    `consistency` can report the gap, but the *performance* has to describe
    something the aeroplane could do. Charging a fraction of the full climb
    instead is what made fuel fall as the target rose: the fraction shrinks
    faster than the climb grows, so asking for more height cost less fuel.

    Once fitted, every target the leg cannot reach charges the same thing --
    which is correct. You get as far as you get, and it costs what it costs.
    """
    if change["distance_nm"] <= geo.distance_nm + _BOUNDARY_EPSILON_NM:
        return change
    reachable = reachable_altitude(
        entry_altitude_ft,
        exit_altitude_ft,
        geo.distance_nm,
        geo,
        aircraft,
        conditions,
    )
    fitted = _solve_change(entry_altitude_ft, reachable, geo, aircraft, conditions)
    return fitted if fitted is not None and fitted["climb"] is not None else change


def _with_level_remainder(
    change: dict,
    geo: Segment,
    conditions: Conditions,
    cruise_point: perf.CruisePoint,
    exit_altitude_ft: float,
) -> tuple[float, float, float]:
    """Fly whatever is left of a climb leg after the climb tops out.

    A climb leg in user-driven mode is as long as the two waypoints make it,
    which is rarely as long as the climb needs. The old behaviour flew the
    whole thing at climb airspeed and pro-rated the published climb fuel over
    it, which charged a 50 nm leg ten times the fuel the POH gives for the
    climb on it -- and, because the pro-rating shrank as the climb grew, made
    fuel *fall* as the target altitude rose.

    Here the climb is the climb and the rest is level flight. Returns the
    airspeed to report for the row, the level ground distance, and the level
    time. The reported airspeed is the single figure that reproduces the two
    halves' total time against the wind, so that a pilot checking
    ground speed = distance / time on the finished row still finds it holds.
    """
    climb_nm = change["distance_nm"]
    climb_minutes = change["climb"].time_min
    rest_nm = geo.distance_nm - climb_nm
    if rest_nm <= _BOUNDARY_EPSILON_NM or climb_minutes <= 0.0:
        return change["tas_kt"], 0.0, 0.0

    level_gs = _ground_speed(
        geo, conditions.winds.at(exit_altitude_ft), cruise_point.ktas
    )
    if level_gs <= 0.0:
        return change["tas_kt"], 0.0, 0.0

    rest_minutes = 60.0 * rest_nm / level_gs
    total_minutes = climb_minutes + rest_minutes
    effective_tas = _tas_for_ground_speed(
        geo,
        conditions.winds.at(0.5 * (change["altitude_ft"] + exit_altitude_ft)),
        60.0 * geo.distance_nm / total_minutes,
    )
    return effective_tas, rest_nm, rest_minutes


def _mark_boundaries(segments: list[ProfileSegment]) -> list[ProfileSegment]:
    """Tag the tops of climb and descent.

    A climb's end is a top of climb only if the climb actually stops there --
    two climb legs in a row have no TOC between them. Symmetrically a descent's
    start is a top of descent only if the descent begins there.

    The role is carried beside the name rather than replacing it, so a charted
    waypoint the pilot chose as their top of climb still prints the name that
    is on the sectional.
    """
    marked: list[ProfileSegment] = []
    for index, segment in enumerate(segments):
        after = segments[index + 1].phase if index + 1 < len(segments) else None
        before = segments[index - 1].phase if index else None
        end_role = "TOC" if segment.phase == "climb" and after != "climb" else None
        start_role = (
            "TOD" if segment.phase == "descent" and before != "descent" else None
        )
        marked.append(replace(segment, start_role=start_role, end_role=end_role))
    return marked


# --- altitude solvers ----------------------------------------------------


def _plateau_altitudes(
    waypoints: list[Waypoint], cruise_altitude_ft: float
) -> list[float]:
    """The altitude to level off at on each leg.

    A waypoint's altitude applies to the leg arriving at it and to every leg
    after, until another waypoint overrides it.
    """
    current = cruise_altitude_ft
    plateaus: list[float] = []
    for _, end in pairwise(waypoints):
        if end.altitude_ft is not None:
            current = end.altitude_ft
        plateaus.append(current)
    return plateaus


def _descent_ceilings(
    waypoints: list[Waypoint],
    plateaus: list[float],
    *,
    destination_elevation: float,
    aircraft: Aircraft,
    conditions: Conditions,
    leg_winds: LegWinds | None = None,
) -> list[float]:
    """The highest altitude each waypoint may be crossed at, walking backwards.

    The destination must be arrived at at field elevation. Working back from
    there, each waypoint may be as high as the next one plus whatever can be
    lost on the leg between them -- capped by the altitude actually planned
    for, since this is a limit and not a target.

    Returns one altitude per waypoint, so `ceilings[i]` applies at
    `waypoints[i]`.
    """
    ceilings = [float("inf")] * len(waypoints)
    ceilings[-1] = destination_elevation

    for index in range(len(waypoints) - 2, -1, -1):
        geo = inverse(waypoints[index].position, waypoints[index + 1].position)
        below = ceilings[index + 1]
        gain = _descendable_ft(
            geo,
            below,
            aircraft,
            _in_leg_wind(conditions, leg_winds, index, "descent"),
        )
        ceilings[index] = min(plateaus[index], below + gain)

    return ceilings


def _descendable_ft(
    geo: Segment, from_altitude_ft: float, aircraft: Aircraft, conditions: Conditions
) -> float:
    """Height that can be shed over a leg at the configured descent rate.

    Ground speed is needed to calculate descent distance, so we need airspeed and 
    wind for this.
    """
    if geo.distance_nm <= 0 or aircraft.descent_rate_fpm <= 0:
        return 0.0
    tas_kt = tas_from_cas(
        aircraft.descent_speed_kias, conditions.density_altitude_ft(from_altitude_ft)
    )
    wind = conditions.winds.at(from_altitude_ft)
    try:
        ground_speed = solve_wind_triangle(
            geo.true_course_deg, tas_kt, wind.from_deg, wind.speed_kt
        ).ground_speed_kt
    except WindTooStrong:
        ground_speed = tas_kt
    if ground_speed <= 0:
        return 0.0
    minutes = 60.0 * geo.distance_nm / ground_speed
    return aircraft.descent_rate_fpm * minutes


def reachable_altitude(
    from_altitude_ft: float,
    toward_altitude_ft: float,
    available_nm: float,
    geo: Segment,
    aircraft: Aircraft,
    conditions: Conditions,
) -> float:
    """Calculate the maximum altitude the airplane can reach, given a ground speed

    Bisected on `_solve_change`, because the POH climb rate falls with altitude
    and the relationship is not linear. Used both when a leg is too short to
    finish its climb, or when a "climb to" states no altitude at all and the
    answer is simply "as high as it gets".
    """
    if available_nm <= 0 or toward_altitude_ft <= from_altitude_ft:
        return from_altitude_ft

    def distance_to(altitude: float) -> float:
        piece = _solve_change(from_altitude_ft, altitude, geo, aircraft, conditions)
        return piece["distance_nm"] if piece else 0.0

    if distance_to(toward_altitude_ft) <= available_nm:
        return toward_altitude_ft

    low, high = from_altitude_ft, toward_altitude_ft
    for _ in range(48):
        middle = 0.5 * (low + high)
        if distance_to(middle) <= available_nm:
            low = middle
        else:
            high = middle
    return low


def _descent_reachable(
    from_altitude_ft: float,
    floor_altitude_ft: float,
    available_nm: float,
    geo: Segment,
    aircraft: Aircraft,
    conditions: Conditions,
) -> float:
    """How low the aeroplane gets in the distance available, at the set rate.

    The mirror of `reachable_altitude`. Bisected rather than solved directly
    because the descent's ground speed depends on the altitude band it is flown
    in, which is what we are solving for. Floored at the arrival elevation: a
    descent that has distance to spare levels off there rather than digging.
    """
    if available_nm <= 0 or floor_altitude_ft >= from_altitude_ft:
        return from_altitude_ft

    def distance_to(altitude: float) -> float:
        piece = _solve_change(from_altitude_ft, altitude, geo, aircraft, conditions)
        return piece["distance_nm"] if piece else 0.0

    if distance_to(floor_altitude_ft) <= available_nm:
        return floor_altitude_ft

    # Distance *decreases* as the target altitude rises -- a shallower descent
    # needs less room -- so the bracket runs the opposite way round to the
    # climb's. `hi` is the shallowest altitude known to fit, `lo` is too deep
    # to fit, and the answer is `hi`: assuming the aeroplane gets down less
    # than hoped is the conservative side of a descent.
    lo, hi = floor_altitude_ft, from_altitude_ft
    for _ in range(48):
        middle = 0.5 * (lo + hi)
        if distance_to(middle) <= available_nm:
            hi = middle
        else:
            lo = middle
    return hi


def _altitude_bands(
    from_altitude_ft: float, to_altitude_ft: float
) -> list[tuple[float, float]]:
    """Split an altitude change into bands aligned on the chart's 1000 ft rows.

    Always returned low-to-high regardless of which way the aeroplane is going,
    because each band is solved independently and only summed. Aligning on
    round altitudes rather than simply dividing the change into equal pieces
    keeps a climb through a given slab of air integrated the same way whatever
    altitude it started from, so two routes that cross 6000 ft agree about what
    that part of the climb cost.
    """
    low, high = sorted((from_altitude_ft, to_altitude_ft))
    edges = [low]
    edge = (math.floor(low / _PROFILE_BAND_FT) + 1.0) * _PROFILE_BAND_FT
    while edge < high - _ALTITUDE_EPSILON_FT:
        edges.append(edge)
        edge += _PROFILE_BAND_FT
    edges.append(high)
    return list(pairwise(edges))


def _ground_speed(geo: Segment, wind: Wind, tas_kt: float) -> float:
    """Ground speed along the leg, or still air where the course cannot be held.

    The leg itself will refuse for the same reason, so falling back here lets
    the profile be laid out and the real error reported against the row rather
    than surfacing as a failure to plan at all.
    """
    try:
        return max(
            0.0,
            solve_wind_triangle(
                geo.true_course_deg, tas_kt, wind.from_deg, wind.speed_kt
            ).ground_speed_kt,
        )
    except WindTooStrong:
        return tas_kt


def _tas_for_ground_speed(geo: Segment, wind: Wind, ground_speed_kt: float) -> float:
    """The true airspeed that would make good `ground_speed_kt` along the leg.

    The inverse of `solve_wind_triangle`'s ground speed. Both the crosswind and
    the along-course component depend only on the wind and the course, so

        GS - along = TAS cos(WCA) = sqrt(TAS^2 - crosswind^2)

    rearranges to a closed form with no iteration. Used to report a marched
    climb or descent as the single airspeed `navlog` re-flies the row at -- see
    `_solve_change`.
    """
    towards = math.radians(wind.from_deg + 180.0 - geo.true_course_deg)
    crosswind = -wind.speed_kt * math.sin(towards)
    along = wind.speed_kt * math.cos(towards)
    return math.hypot(max(0.0, ground_speed_kt - along), crosswind)


def _solve_change(
    from_altitude_ft: float,
    to_altitude_ft: float,
    geo: Segment,
    aircraft: Aircraft,
    conditions: Conditions,
) -> dict | None:
    """Time, airspeed and ground distance for one altitude change.

    `None` when there is no change worth flying, which is what lets the caller
    treat "climb then level" and "level all the way" as one shape.

    The change is **integrated band by band** rather than solved once at its
    midpoint. Airspeed, wind and temperature all vary with altitude, and a
    single midpoint sample cannot see any of it: in a forecast running 5 kt at
    3000 ft to 45 kt at 9000 ft, a sea-level-to-9500 ft climb comes out 1.5 nm
    short, and 2.5 nm long if the shear runs the other way. That is where the
    top of climb lands, so it decides how much of the leg is charged at climb
    rate rather than cruise rate.

    Time and fuel are unaffected by the marching on a standard day -- the POH
    columns are cumulative, so differencing them telescopes and the bands sum
    back to the same total. It is the *ground* distance the integration moves,
    because that is what the wind acts on: the published climb distance is
    never read, the band's POH climb speed is flown against the band's wind
    and the ground distance falls out of that.

    `tas_kt` comes back as the single airspeed that reproduces the marched
    ground speed against the wind at the reported altitude, so that `navlog`
    re-flying the row from `distance_nm` recovers the time this integrated.
    """
    if abs(to_altitude_ft - from_altitude_ft) <= _ALTITUDE_EPSILON_FT:
        return None
    phase = "climb" if to_altitude_ft > from_altitude_ft else "descent"
    mid_altitude = (from_altitude_ft + to_altitude_ft) / 2.0

    total_minutes = 0.0
    total_distance = 0.0
    climb_time = climb_fuel = climb_kias_minutes = 0.0

    for band_low, band_high in _altitude_bands(from_altitude_ft, to_altitude_ft):
        band_mid = 0.5 * (band_low + band_high)
        if phase == "climb":
            # Time and fuel come from differencing the cumulative table at
            # the band's ends; the climb speed it reports is the average of
            # the two rows, which is what this band is flown at below.
            piece = perf.climb_from_to(
                conditions.pressure_altitude_ft(band_low),
                conditions.pressure_altitude_ft(band_high),
                oat_c=conditions.oat_c(band_mid),
            )
            minutes = piece.time_min
            climb_time += piece.time_min
            climb_fuel += piece.fuel_gal
            climb_kias_minutes += piece.kias * piece.time_min
            tas_kt = tas_from_cas(piece.kias, conditions.density_altitude_ft(band_mid))
        else:
            minutes = (band_high - band_low) / aircraft.descent_rate_fpm
            tas_kt = tas_from_cas(
                aircraft.descent_speed_kias,
                conditions.density_altitude_ft(band_mid),
            )
        total_minutes += minutes
        total_distance += (
            _ground_speed(geo, conditions.winds.at(band_mid), tas_kt) * minutes / 60.0
        )

    climb = (
        perf.ClimbSegment(
            climb_time,
            climb_fuel,
            climb_kias_minutes / climb_time if climb_time > 0 else 0.0,
        )
        if phase == "climb"
        else None
    )
    if total_minutes <= 0.0:
        return None
    return {
        "phase": phase,
        "altitude_ft": mid_altitude,
        "tas_kt": _tas_for_ground_speed(
            geo,
            conditions.winds.at(mid_altitude),
            60.0 * total_distance / total_minutes,
        ),
        "climb": climb,
        "distance_nm": max(0.0, total_distance),
    }
