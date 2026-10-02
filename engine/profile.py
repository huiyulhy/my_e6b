"""The vertical profile: what each leg of a route does about altitude.

A route is a list of waypoints. This module turns it into a list of
`ProfileSegment` -- **exactly one per leg** -- each of which is a climb, a
cruise or a descent at a single altitude, airspeed and fuel flow. 

Two ways to get there, sharing one model:

* **User-driven.** The pilot declares each leg's `segment_type`. Where no
  altitude is given, it calculates the maximum altitude reachable based on POH
  ,same for descend at the configured rate.
* **Hybrid.** Legs are declared `automatic`; `resolve_route` expands them
  against a target cruise altitude, **inserting TOC and TOD waypoints into the
  route** so that afterwards every leg is concrete and the user-driven walk
  runs unchanged. A drawn leg may also carry `VerticalEvent`s -- "start the
  climb here", "be level by here" -- which pin one end of an altitude change
  to a place and let the other end float with the wind (BOC/TOC, BOD/TOD).
  A leg with no events is flown exactly as the old automatic mode flew it.
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
PLANNING_MODES: tuple[str, ...] = ("manual", "hybrid")
# Hybrid grew out of the automatic mode and plans a leg with no events exactly
# as it did, so the old name is still accepted -- saved missions and callers
# written against it keep working.
_MODE_ALIASES = {"auto": "hybrid", "automatic": "hybrid"}

EVENT_KINDS: tuple[str, ...] = ("start", "complete")
# An event clicked further than this off its leg is probably on the wrong one.
_EVENT_OFF_TRACK_NM = 2.0

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


def normalise_planning_mode(mode: str) -> str:
    """The canonical name of a planning mode, or `RouteError` if there is none."""
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in PLANNING_MODES:
        raise RouteError(
            f"unknown planning mode {mode!r}; "
            f"expected one of {', '.join(PLANNING_MODES)}"
        )
    return mode


@dataclass(frozen=True)
class VerticalEvent:
    """An altitude change pinned to a place on a drawn leg.

    * `start`: hold altitude until here, then change to the target. The bottom
      of the change (BOC/TOD) is pinned; its top (TOC/BOD) floats downstream
      by however far the change takes in the wind.
    * `complete`: be at the target by here. The end of the change is pinned and
      its beginning floats upstream.

    Whether it is a climb or a descent follows from the target against the
    altitude the aeroplane arrives with, so one shape covers all four points.

    Kept as a *place* rather than a distance along the leg: the reason for it
    is usually on the chart -- the edge of a Class B shelf -- and it should stay
    there if a fix at either end of the leg is moved.
    """

    kind: str  # start | complete
    position: LatLon
    target_altitude_ft: float


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

    # A stable identity for a pilot's waypoint, so that what is said about the
    # leg between two of them -- its events, a typed wind -- survives the
    # route being edited around it. `None` for callers that never key anything.
    id: str | None = None

    # Altitude changes pinned to places on the leg arriving here. Hybrid mode
    # only; user-driven mode declares its legs outright.
    events: tuple[VerticalEvent, ...] = ()

    # On a generated point, the drawn leg it was inserted into -- see
    # `segment_key`. `None` on the pilot's own points.
    segment_key: str | None = None

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
    # "TOC" / "TOD" / "BOC" / "BOD" when this segment's boundary is a top or
    # bottom of climb or descent.
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


def segment_key(start: Waypoint, end: Waypoint) -> str | None:
    """The name of the drawn leg from `start` to `end`: `"<id>><id>"`.

    `None` unless both ends carry an id. This is what the pilot's statements
    about a leg are filed under, rather than a navlog row, because the rows a
    leg is cut into change every time the wind moves a top of climb.
    """
    if start.id is None or end.id is None:
        return None
    return f"{start.id}>{end.id}"


def pattern_altitude(waypoint: Waypoint, default_pattern_agl_ft: float = 1000.0) -> float:
    """A field's traffic pattern altitude, MSL.

    Its published pattern height, or `default_pattern_agl_ft`, above its
    elevation, rounded to the nearest 100 ft as TPAs are published: KMOD at
    99 ft comes out at 1,100.
    """
    agl = (
        waypoint.pattern_altitude_agl_ft
        if waypoint.pattern_altitude_agl_ft is not None
        else default_pattern_agl_ft
    )
    return round(((waypoint.elevation_ft or 0.0) + agl) / 100.0) * 100.0


def arrival_altitude(waypoint: Waypoint, default_pattern_agl_ft: float = 1000.0) -> float:
    """The altitude a flight arrives over a field it lands at.

    The descent ends at traffic pattern altitude, not on the runway -- the
    pattern is its own row, and the landing is part of it. A crossing altitude
    the pilot set on the field wins: a published TPA that is not a round
    1,000 ft, or an overhead join above the pattern.
    """
    if waypoint.altitude_ft is not None:
        return waypoint.altitude_ft
    return pattern_altitude(waypoint, default_pattern_agl_ft)


class PhaseNamer:
    """Names the inserted points: TOC, TOD, BOC, BOD, then TOC2, TOD2 ...

    `role` is what the point is: the top or bottom of a climb or descent.
    """

    _ROLES = ("TOC", "TOD", "BOC", "BOD")

    def __init__(self) -> None:
        self._counts = dict.fromkeys(self._ROLES, 0)

    def next(self, phase_or_role: str) -> str:
        role = {"climb": "TOC", "descent": "TOD"}.get(phase_or_role, phase_or_role)
        self._counts[role] += 1
        n = self._counts[role]
        return role if n == 1 else f"{role}{n}"


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
    mode = normalise_planning_mode(mode)

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
    ceilings, caps = _descent_ceilings(
        waypoints,
        plateaus,
        destination_elevation=destination_elevation,
        aircraft=aircraft,
        conditions=conditions,
        leg_winds=leg_winds,
    )

    resolved: list[Waypoint] = [waypoints[0]]
    entry = departure_elevation
    arriving = "cruise"
    for index, (start, end) in enumerate(pairwise(waypoints)):
        # Never aim higher than the point from which the rest of the route can
        # still be flown down.
        plateau = min(plateaus[index], caps[index])
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
            from_the_ground=index == 0,
            # A climb still going at the last waypoint carries on through it.
            climbing_in=arriving == "climb",
        )
        arriving = end_type
        resolved.extend(inserted)
        # A crossing altitude the aeroplane cannot make is the pilot's to hear
        # about, not the planner's to quietly lower. The rows are built to the
        # declared altitude either way (see `build_segments`), so planning on
        # from it is what keeps the legs after it continuous with them.
        if end.altitude_ft is not None and abs(entry - end.altitude_ft) > 50.0:
            verb = "climb" if end.altitude_ft > entry else "descend"
            warnings.append(
                f"The aeroplane cannot {verb} to {end.altitude_ft:,.0f} ft by "
                f"{end.name}; by the POH it gets to {entry:,.0f} ft there. "
                f"Start the {'climb' if verb == 'climb' else 'descent'} earlier "
                f"or move the waypoint."
            )
            entry = end.altitude_ft
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
    from_the_ground: bool = False,
    climbing_in: bool = False,
) -> tuple[list[Waypoint], str, float]:
    """Split one leg, returning the points to insert before `end`.

    Also returns the phase of the piece that *arrives* at `end`, and the
    altitude actually reached there -- which is not always the one asked for.
    A waypoint dropped three miles into a fourteen mile climb leaves the
    aeroplane part way up, and the caller carries that forward so the climb
    continues on the next leg and the top of climb lands where it really falls.

    The leg is walked from its start. The pilot's events on it come first, in
    the order they fall along it, each pinning one end of an altitude change
    to a place; the aeroplane holds its altitude between them. Whatever is
    left of the leg after the last event is laid out as the automatic planner
    always has: a climb to the plateau out of the cursor, a descent to the
    exit into the end, and level flight in between. A leg with no events is
    therefore exactly the automatic leg.

    `from_the_ground` is the first leg of a flight, which starts on the
    runway. There is no holding altitude there, so a "be level by here" is
    climbed to straight away rather than as late as possible.
    """
    geo = inverse(start.position, end.position)
    if geo.distance_nm <= 0:
        return [], "cruise", entry_altitude_ft

    walk = _LegWalk(
        start=start,
        end=end,
        geo=geo,
        altitude_ft=entry_altitude_ft,
        aircraft=aircraft,
        conditions=conditions,
        winds_by_phase=winds_by_phase,
        warnings=warnings,
        on_the_ground=from_the_ground,
        climbing_in=climbing_in,
    )
    for along_nm, event in _events_along(start, end, geo, warnings):
        if not walk.fly_event(along_nm, event):
            # The change ran off the end of the leg; the next leg finishes it.
            return walk.finish(names)

    # With events, the plateau is where the last of them left the aeroplane
    # unless the pilot asked for more at the end of the leg; the planner never
    # climbs past what the pilot pinned.
    plateau = plateau_altitude_ft
    walk.fly_remainder(plateau, min(exit_altitude_ft, max(plateau, walk.altitude_ft)))
    walk.check_holds()
    return walk.finish(names)


# The role of the point that ends a piece of a leg, by the piece's phase and
# the phase that follows it.
_HANDOVER_ROLES = {
    ("climb", "cruise"): "TOC",
    ("climb", "descent"): "TOC",
    ("cruise", "climb"): "BOC",
    ("cruise", "descent"): "TOD",
    ("descent", "cruise"): "BOD",
    ("descent", "climb"): "BOD",
}


class _LegWalk:
    """One drawn leg, cut into pieces from its start to its end.

    Each piece is `(phase, ends_at_nm, altitude_at_end_ft)`. The cursor is the
    distance flown so far and the altitude the aeroplane is at there.
    """

    def __init__(
        self,
        *,
        start: Waypoint,
        end: Waypoint,
        geo: Segment,
        altitude_ft: float,
        aircraft: Aircraft,
        conditions: Conditions,
        winds_by_phase: dict[str, object] | None,
        warnings: list[str],
        on_the_ground: bool = False,
        climbing_in: bool = False,
    ) -> None:
        self.start, self.end, self.geo = start, end, geo
        self.aircraft, self.conditions = aircraft, conditions
        self.winds_by_phase, self.warnings = winds_by_phase, warnings
        self.cursor_nm = 0.0
        self.altitude_ft = altitude_ft
        self.pieces: list[tuple[str, float, float]] = []
        # Still on the runway: nothing has been flown yet on a departure leg.
        self._on_the_ground = on_the_ground
        # The leg starts part way up a climb carried in from the last one, so
        # that climb's first stretch here is not a bottom of climb.
        self._climbing_in = climbing_in
        # "Start here" events that asked for the altitude already held:
        # (distance along, altitude). Checked once the leg is laid out.
        self._holds: list[tuple[float, float]] = []

    # --- the pieces ------------------------------------------------------

    def _change(self, to_altitude_ft: float) -> tuple[dict | None, Conditions]:
        conditions = _in_phase_wind(
            self.conditions, self.winds_by_phase, self.altitude_ft, to_altitude_ft
        )
        return (
            _solve_change(
                self.altitude_ft, to_altitude_ft, self.geo, self.aircraft, conditions,
                open_start=self._climbing_in,
            ),
            conditions,
        )

    def _reach(
        self, toward_ft: float, available_nm: float, conditions: Conditions
    ) -> float:
        """How far toward `toward_ft` the aeroplane gets in `available_nm`."""
        if toward_ft > self.altitude_ft:
            return reachable_altitude(
                self.altitude_ft, toward_ft, available_nm, self.geo, self.aircraft,
                conditions, open_start=self._climbing_in,
            )
        return _descent_reachable(
            self.altitude_ft, toward_ft, available_nm, self.geo, self.aircraft, conditions
        )

    def _level_until(self, along_nm: float) -> None:
        if along_nm - self.cursor_nm > _BOUNDARY_EPSILON_NM:
            self.pieces.append(("cruise", along_nm, self.altitude_ft))
            self._climbing_in = False
        self.cursor_nm = max(self.cursor_nm, along_nm)

    def _changed(self, phase: str, along_nm: float, altitude_ft: float) -> None:
        self.pieces.append((phase, along_nm, altitude_ft))
        self.cursor_nm, self.altitude_ft = along_nm, altitude_ft
        self._on_the_ground = False
        self._climbing_in = False

    # --- the pilot's events ----------------------------------------------

    def fly_event(self, along_nm: float, event: VerticalEvent) -> bool:
        """Fly one event. False if its change is still going at the leg's end."""
        piece, conditions = self._change(event.target_altitude_ft)
        leg = f"{self.start.name} to {self.end.name}"
        late = along_nm < self.cursor_nm - _BOUNDARY_EPSILON_NM
        if piece is None and event.kind == "complete" and late:
            # At the target now, but only since the cursor: the change before
            # this event runs past its point. A tailwind stretching the climb
            # out of an earlier "start here" is the usual way in.
            self.warnings.append(
                f"On {leg}, the aeroplane is not at "
                f"{event.target_altitude_ft:,.0f} ft by the point {along_nm:.1f} nm "
                f"along: the change before it gets there "
                f"{self.cursor_nm - along_nm:.1f} nm later. Start it earlier or "
                f"move the point."
            )
            return True
        if piece is None:
            # Already at the target. A "start here" still means "not before
            # here": hold to the point, and let the rest of the leg climb or
            # descend from there, toward wherever the leg is headed -- the
            # usual way to say "start the climb to the next fix's altitude
            # here". Only if nothing follows is it worth a word; see
            # `check_holds`.
            if event.kind == "start":
                self._level_until(min(along_nm, self.geo.distance_nm))
                self._holds.append((along_nm, event.target_altitude_ft))
            return True
        phase, needed = piece["phase"], piece["distance_nm"]
        length = self.geo.distance_nm

        if along_nm < self.cursor_nm - _BOUNDARY_EPSILON_NM:
            self.warnings.append(
                f"On {leg}, the {_event_label(event, phase)} {along_nm:.1f} nm along "
                f"falls inside the altitude change before it, which ends "
                f"{self.cursor_nm:.1f} nm along."
            )

        if event.kind == "start":
            if self._on_the_ground:
                self.warnings.append(
                    f"On {leg}, the {_event_label(event, phase)} {along_nm:.1f} nm "
                    f"along would hold field elevation until then. Add a "
                    f"level-off before it for the altitude to climb out to."
                )
            begin = max(along_nm, self.cursor_nm)
            self._level_until(begin)
            if begin + needed > length + _BOUNDARY_EPSILON_NM:
                reached = self._reach(event.target_altitude_ft, length - begin, conditions)
                self._changed(phase, length, reached)
                return False
            self._changed(phase, begin + needed, event.target_altitude_ft)
            return True

        # complete: the end of the change is pinned, its beginning floats --
        # except off the runway, where the climb starts at once and the point
        # is where it must be done by.
        along_nm = max(along_nm, self.cursor_nm)
        begin = along_nm - needed
        if self._on_the_ground and begin >= self.cursor_nm:
            self._changed(phase, needed, event.target_altitude_ft)
            return True
        if begin < self.cursor_nm - _BOUNDARY_EPSILON_NM:
            reached = self._reach(
                event.target_altitude_ft, along_nm - self.cursor_nm, conditions
            )
            self.warnings.append(
                f"On {leg}, the aeroplane cannot {phase} to "
                f"{event.target_altitude_ft:,.0f} ft by the point {along_nm:.1f} nm "
                f"along; it is {self.cursor_nm - begin:.1f} nm short and gets to "
                f"{reached:,.0f} ft there."
            )
            self._changed(phase, along_nm, reached)
            return True
        self._level_until(begin)
        self._changed(phase, along_nm, event.target_altitude_ft)
        return True

    # --- what is left of the leg -----------------------------------------

    def fly_remainder(self, plateau_ft: float, exit_ft: float) -> None:
        """Climb to the plateau out of the cursor, descend to the exit into
        the end, and fly level in between -- the automatic leg.

        The rule for what the planner adds on its own: at most three pieces,
        a change, level, a change, and never two changes the same way with
        level between them. A stepped climb or descent is the pilot's to ask
        for with events, which are flown before this and are not merged.
        """
        available = self.geo.distance_nm - self.cursor_nm
        lead, lead_conditions = self._change(plateau_ft)
        entry = self.altitude_ft
        self.altitude_ft = plateau_ft
        tail, _ = self._change(exit_ft)
        self.altitude_ft = entry

        if lead is None and tail is None:
            self._level_until(self.geo.distance_nm)
            if not self.pieces:
                self.altitude_ft = plateau_ft
            return

        # Down to the plateau, level, then down again is one descent with a
        # step in it that nobody would fly. It happens where the aeroplane
        # arrives above the most this leg can lose; descend once, as late as
        # the exit allows, and let `_fly_straight_to` say if it cannot fit.
        if (
            lead is not None
            and tail is not None
            and lead["phase"] == "descent"
            and tail["phase"] == "descent"
        ):
            self._fly_straight_to(exit_ft)
            return

        # The leg is still changing altitude when it ends: it runs out of
        # distance before the plateau, and there is no descent to fit in
        # afterwards. Fly as much of the change as fits and hand the rest to
        # the next leg. This is the ordinary case for a waypoint placed inside
        # the climb, so it is not a warning -- nothing is wrong with the route.
        if lead is not None and tail is None and lead["distance_nm"] > available:
            reached = self._reach(plateau_ft, available, lead_conditions)
            if abs(reached - entry) <= _ALTITUDE_EPSILON_FT:
                self._level_until(self.geo.distance_nm)
                return
            self._changed(lead["phase"], self.geo.distance_nm, reached)
            return

        needed = sum(p["distance_nm"] for p in (lead, tail) if p)
        if needed > available:
            # Up to the plateau and back down does not fit. Never squeeze it
            # in by shrinking the distances and keeping the altitudes: that
            # draws a climb the POH says cannot be flown (2,500 fpm out of
            # Palo Alto). Nor climb above where the leg ends only to come back
            # down -- odd flying, and over a "cross at" below a Class B shelf
            # it is the shelf. Change straight toward the exit instead, as far
            # as the performance allows.
            self._fly_straight_to(exit_ft)
            return

        if lead:
            self._changed(lead["phase"], self.cursor_nm + lead["distance_nm"], plateau_ft)
        else:
            self.altitude_ft = plateau_ft
        if tail:
            self._level_until(self.geo.distance_nm - tail["distance_nm"])
            self._changed(tail["phase"], self.geo.distance_nm, exit_ft)
        else:
            self._level_until(self.geo.distance_nm)

    def _fly_straight_to(self, exit_ft: float) -> None:
        """Change from the cursor toward `exit_ft`, level once there.

        A climb that runs out of leg carries on into the next, as any climb
        does. A descent that runs out of leg arrives high, which the pilot is
        told: the next constraint may not be met.
        """
        available = self.geo.distance_nm - self.cursor_nm
        change, conditions = self._change(exit_ft)
        if change is None:
            self._level_until(self.geo.distance_nm)
            return
        if change["distance_nm"] <= available:
            if change["phase"] == "climb":
                self._changed("climb", self.cursor_nm + change["distance_nm"], exit_ft)
                self._level_until(self.geo.distance_nm)
            else:
                self._level_until(self.geo.distance_nm - change["distance_nm"])
                self._changed("descent", self.geo.distance_nm, exit_ft)
            return
        reached = self._reach(exit_ft, available, conditions)
        if change["phase"] == "descent":
            self.warnings.append(
                f"{self.start.name} to {self.end.name}: the descent to "
                f"{exit_ft:,.0f} ft needs {change['distance_nm']:.1f} nm at "
                f"{self.aircraft.descent_rate_fpm:.0f} fpm but only "
                f"{available:.1f} nm is left; the airplane reaches "
                f"{self.end.name} at {reached:,.0f} ft. Descend sooner or faster."
            )
        if abs(reached - self.altitude_ft) <= _ALTITUDE_EPSILON_FT:
            self._level_until(self.geo.distance_nm)
            return
        self._changed(change["phase"], self.geo.distance_nm, reached)

    def check_holds(self) -> None:
        """Say so where a "start here" held its altitude and nothing followed.

        Holding to the point and then climbing toward the leg's own altitude
        is what the event is for, and needs no comment. Holding to the end of
        the leg with no change at all means the event did nothing the pilot
        could have wanted.
        """
        leg = f"{self.start.name} to {self.end.name}"
        for along_nm, altitude_ft in self._holds:
            changes_after = any(
                phase != "cruise" and ends_at > along_nm + _BOUNDARY_EPSILON_NM
                for phase, ends_at, _ in self.pieces
            )
            if not changes_after:
                self.warnings.append(
                    f"On {leg}, the start-of-climb event {along_nm:.1f} nm along "
                    f"asks for {altitude_ft:,.0f} ft, which is where the aeroplane "
                    f"already is, and nothing after it changes altitude. Set the "
                    f"event to the altitude to climb or descend to."
                )

    # --- the result ------------------------------------------------------

    def finish(self, names: PhaseNamer) -> tuple[list[Waypoint], str, float]:
        """The points to insert, the arriving phase, and the altitude reached."""
        merged: list[tuple[str, float, float]] = []
        for piece in self.pieces:
            # Two pieces in the same phase back to back -- a climb resumed at
            # the point the last one stopped -- are one piece of flying.
            if merged and merged[-1][0] == piece[0]:
                merged[-1] = piece
            else:
                merged.append(piece)
        if not merged:
            return [], "cruise", self.altitude_ft

        key = segment_key(self.start, self.end)
        inserted: list[Waypoint] = []
        for (phase, ends_at, altitude), (following, _, _) in pairwise(merged):
            inserted.append(
                Waypoint(
                    name=names.next(_HANDOVER_ROLES[(phase, following)]),
                    position=self.geo.point_at_nm(ends_at),
                    kind="phase",
                    # The altitude is carried on the point itself, so
                    # rebuilding the profile from the resolved route
                    # reproduces it exactly.
                    altitude_ft=altitude,
                    segment_type=phase,
                    generated=True,
                    segment_key=key,
                )
            )
        phase, _, altitude = merged[-1]
        return inserted, phase, altitude


def _event_label(event: VerticalEvent, phase: str) -> str:
    if event.kind == "start":
        return f"start of {phase}"
    return f"level-off at {event.target_altitude_ft:,.0f} ft"


def _events_along(
    start: Waypoint, end: Waypoint, geo: Segment, warnings: list[str]
) -> list[tuple[float, VerticalEvent]]:
    """The events on a leg as `(distance along it, event)`, in flying order.

    Each is projected onto the leg where it is flown. One that projects off
    either end is held at that end, and one well to the side of the leg is
    most likely on a different leg; both are said out loud rather than
    silently moved.
    """
    placed: list[tuple[float, VerticalEvent]] = []
    for event in end.events:
        if event.kind not in EVENT_KINDS:
            raise RouteError(
                f"unknown event {event.kind!r} on {start.name} to {end.name}; "
                f"expected one of {', '.join(EVENT_KINDS)}"
            )
        along = geo.along_track_nm(event.position)
        off = abs(geo.cross_track_nm(event.position))
        if off > _EVENT_OFF_TRACK_NM:
            warnings.append(
                f"An event on {start.name} to {end.name} is {off:.1f} nm off "
                f"the leg; it is flown at the point abeam it."
            )
        if along < 0.0 or along > geo.distance_nm:
            warnings.append(
                f"An event on {start.name} to {end.name} lies beyond the end "
                f"of the leg; it is flown at the nearer end."
            )
            along = min(max(along, 0.0), geo.distance_nm)
        placed.append((along, event))
    return sorted(placed, key=lambda item: item[0])


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
    arrival_altitude_ft: float | None = None,
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
        # A climb that carries on through either end of this leg is read
        # mid-climb there, not rounded to the book's row.
        open_start = phase == "climb" and bool(segments) and segments[-1].phase == "climb"
        open_end = (
            phase == "climb"
            and index + 2 < len(waypoints)
            and waypoints[index + 2].segment_type == "climb"
        )
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
                open_start=open_start,
                # The last leg descends to the arrival altitude, the pattern,
                # rather than all the way to the runway.
                floor_ft=(
                    arrival_altitude_ft if index == len(waypoints) - 2 else None
                ),
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
                open_start=open_start,
                open_end=open_end,
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
    open_start: bool = False,
    floor_ft: float | None = None,
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
            open_start=open_start,
        )
    if floor_ft is None:
        floor_ft = end.elevation_ft if end.elevation_ft is not None else 0.0
    return _descent_reachable(
        entry_altitude_ft,
        floor_ft,
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
    open_start: bool = False,
    open_end: bool = False,
) -> ProfileSegment:
    """Airspeed and POH climb figures for one declared leg."""
    mid = 0.5 * (entry_altitude_ft + exit_altitude_ft)
    change = _solve_change(
        entry_altitude_ft, exit_altitude_ft, geo, aircraft, conditions,
        open_start=open_start, open_end=open_end,
    )
    level_nm = level_minutes = 0.0

    if phase == "climb" and change is not None and change["phase"] == "climb":
        change = _fitted_to_the_leg(
            change, entry_altitude_ft, exit_altitude_ft, geo, aircraft, conditions,
            open_start=open_start,
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
    open_start: bool = False,
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
        open_start=open_start,
    )
    fitted = _solve_change(
        entry_altitude_ft, reachable, geo, aircraft, conditions,
        open_start=open_start, open_end=True,
    )
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
    """Tag the tops and bottoms of climb and descent.

    A level leg that hands over to a climb ends at a bottom of climb, and a
    descent that levels off (or turns back into a climb) ends at a bottom of
    descent. A descent into the destination has no bottom: it lands.

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
        end_role = None
        if segment.phase == "climb" and after != "climb":
            end_role = "TOC"
        elif segment.phase == "cruise" and after == "climb":
            end_role = "BOC"
        elif segment.phase == "descent" and after in ("cruise", "climb"):
            end_role = "BOD"
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
    after, until another waypoint overrides it. So does the target of the last
    event on a leg.
    """
    current = cruise_altitude_ft
    plateaus: list[float] = []
    last = len(waypoints) - 1
    for index, (start, end) in enumerate(pairwise(waypoints)):
        # An event's target holds from where it is flown onward, as a
        # crossing altitude does; the last one on the leg is what is left.
        if end.events:
            geo = inverse(start.position, end.position)
            current = _events_along(start, end, geo, [])[-1][1].target_altitude_ft
        # The landing field's altitude is where the descent ends, not an
        # altitude to fly the last leg at.
        if end.altitude_ft is not None and index + 1 < last:
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
) -> tuple[list[float], list[float]]:
    """The highest altitude each waypoint may be crossed at, walking backwards.

    The destination must be arrived at at `destination_elevation` -- which is
    the arrival altitude over the field, the pattern altitude when called from
    `navlog`, field elevation for a caller that lands on it. Working back from
    there, each waypoint may be as high as the next one plus whatever can be
    lost on the leg between them -- capped by the altitude actually planned
    for, since this is a limit and not a target.

    Returns two lists, one altitude per waypoint:

    * `ceilings[i]`, the altitude to be at over `waypoints[i]` -- what the leg
      arriving there descends to, and what the leg before it is planned down
      toward. A crossing altitude the pilot set replaces it.
    * `caps[i]`, the highest the leg *leaving* `waypoints[i]` may go and still
      get down to `ceilings[i + 1]` by its end. Never a crossing altitude: a
      "cross VPKGO at 2500" says where the aeroplane is at VPKGO, not that it
      must stay there all the way to the next fix.
    """
    ceilings = [float("inf")] * len(waypoints)
    caps = [float("inf")] * len(waypoints)
    ceilings[-1] = destination_elevation

    for index in range(len(waypoints) - 2, -1, -1):
        geo = inverse(waypoints[index].position, waypoints[index + 1].position)
        below = ceilings[index + 1]
        gain = _descendable_ft(
            geo,
            below,
            aircraft,
            _in_leg_wind(conditions, leg_winds, index, "descent"),
            up_to_ft=plateaus[index],
        )
        caps[index] = ceilings[index] = min(plateaus[index], below + gain)
        # A crossing altitude the pilot set is theirs, not a limit to plan
        # under: planning to cross lower so the next descent fits would be
        # overruled when the rows are built, and leave a descent of no height.
        # If what follows cannot be flown from it, the forward pass says so.
        if index > 0 and waypoints[index].altitude_ft is not None:
            ceilings[index] = waypoints[index].altitude_ft

    return ceilings, caps


def _descendable_ft(
    geo: Segment,
    from_altitude_ft: float,
    aircraft: Aircraft,
    conditions: Conditions,
    up_to_ft: float | None = None,
) -> float:
    """Height that can be shed over a leg at the configured descent rate.

    Solved with the same band-by-band descent the forward pass flies
    (`_solve_change`), so the two passes agree on where a descent has to
    start. A single sample at the bottom of the descent -- the airspeed and
    wind at `from_altitude_ft` alone -- disagreed with it by a few hundred feet
    in shear, and the forward pass made up the difference with a second, tiny
    descent at the start of the next leg.

    `up_to_ft` bounds the search: nothing above the leg's own plateau is ever
    used, and a long leg's estimate would otherwise reach past the top of the
    atmosphere model.
    """
    if geo.distance_nm <= 0 or aircraft.descent_rate_fpm <= 0:
        return 0.0
    estimate = _descendable_ft_at_the_bottom(geo, from_altitude_ft, aircraft, conditions)
    if estimate <= 0.0:
        return 0.0

    def distance_from(top_ft: float) -> float:
        piece = _solve_change(top_ft, from_altitude_ft, geo, aircraft, conditions)
        return piece["distance_nm"] if piece else 0.0

    high = 2.0 * estimate + 1000.0
    if up_to_ft is not None:
        high = min(high, max(0.0, up_to_ft - from_altitude_ft))
    if high <= 0.0:
        return 0.0
    if distance_from(from_altitude_ft + high) <= geo.distance_nm:
        return high
    low = 0.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        if distance_from(from_altitude_ft + middle) <= geo.distance_nm:
            low = middle
        else:
            high = middle
    return low


def _descendable_ft_at_the_bottom(
    geo: Segment, from_altitude_ft: float, aircraft: Aircraft, conditions: Conditions
) -> float:
    """The quick estimate: airspeed and wind read at the bottom only."""
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
    *,
    open_start: bool = False,
) -> float:
    """Calculate the maximum altitude the airplane can reach, given a ground speed

    Bisected on `_solve_change`, because the POH climb rate falls with altitude
    and the relationship is not linear. Used both when a leg is too short to
    finish its climb, or when a "climb to" states no altitude at all and the
    answer is simply "as high as it gets".

    Reaching the target is a top of climb, and costs the book's rounded time
    to it. Falling short is not: the climb is still going where the distance
    runs out, so that end is read mid-climb. `open_start` says the climb was
    already under way where this distance begins.
    """
    if available_nm <= 0 or toward_altitude_ft <= from_altitude_ft:
        return from_altitude_ft

    def distance_to(altitude: float, *, open_end: bool) -> float:
        piece = _solve_change(
            from_altitude_ft, altitude, geo, aircraft, conditions,
            open_start=open_start, open_end=open_end,
        )
        return piece["distance_nm"] if piece else 0.0

    if distance_to(toward_altitude_ft, open_end=False) <= available_nm:
        return toward_altitude_ft

    low, high = from_altitude_ft, toward_altitude_ft
    for _ in range(48):
        middle = 0.5 * (low + high)
        if distance_to(middle, open_end=True) <= available_nm:
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
    *,
    open_start: bool = False,
    open_end: bool = False,
) -> dict | None:
    """Time, airspeed and ground distance for one altitude change.

    A climb's time is the book's, rounded out to the printed rows at the
    climb's real start and top only (see `perf.climb_from_to`). `open_start`
    and `open_end` say this change begins or ends mid-climb -- at a waypoint
    the climb carries on through -- where it is interpolated instead, as it
    is at every band edge inside the change.

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

    bands = _altitude_bands(from_altitude_ft, to_altitude_ft)
    for band_index, (band_low, band_high) in enumerate(bands):
        band_mid = 0.5 * (band_low + band_high)
        if phase == "climb":
            # Time and fuel come from differencing the cumulative table at
            # the band's ends; the climb speed it reports is the average of
            # the two rows, which is what this band is flown at below.
            piece = perf.climb_from_to(
                conditions.pressure_altitude_ft(band_low),
                conditions.pressure_altitude_ft(band_high),
                oat_c=conditions.oat_c(band_mid),
                round_from=band_index == 0 and not open_start,
                round_to=band_index == len(bands) - 1 and not open_end,
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
