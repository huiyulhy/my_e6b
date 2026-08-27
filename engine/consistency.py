"""Does this navigation log contradict itself?

Everything else in the engine checks inputs *before* it computes: a route with
no elevation is refused, a query outside the POH envelope raises. Nothing looks
at a **finished** navlog and asks whether it hangs together -- whether the
altitude one leg ends at is the one the next starts from, whether a leg
declared a climb actually climbs, whether the descent arrives at the field.

That is what this module does, and it is deliberately separate from building.
`engine.profile.build_segments` emits exactly what the pilot declared, even
when the declaration is self-contradictory, because a planner that quietly
corrects you hides the mistake instead of showing it. This is where the mistake
gets shown.

It is also separate from `engine.preflight`, which stays the authority on the
go/no-go decision -- runway lengths and fuel reserves. That looks only at
airports; this looks only at the profile and the bookkeeping. The two are
presented side by side rather than merged, so it is always clear whether the
aeroplane cannot do something or the *plan* merely says two different things.

Adding a check
--------------
Write `_check_<thing>(ctx) -> list[Finding]` and add it to `_CHECKS`. Each one
is independent and gets the whole log, so they can be tested one at a time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from engine import performance as perf
from engine.geo import inverse

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from engine.navlog import Aircraft, Conditions, Navlog

# Two rows agreeing on an altitude to within this are agreeing.
ALTITUDE_TOLERANCE_FT = 1.0

# A climb may top out this far above what the POH can actually deliver in the
# leg's distance before it counts as a contradiction. It absorbs the bisection's
# own resolution and the fact that a pilot rounds an altitude to the hundred.
CLIMB_TOLERANCE_FT = 200.0

# The steepest descent the plan may imply. `Aircraft.descent_rate_fpm` is what
# the planner *assumes* when it lays a descent out; this is the ceiling on what
# a finished row is allowed to demand.
MAX_DESCENT_RATE_FPM = 500.0
# An automatic descent is laid out at exactly the assumed rate, and the row's
# ETE is then re-derived from distance over ground speed -- so the implied rate
# lands a hair either side of 500. Without this, every ordinary descent flags.
_DESCENT_RATE_EPSILON_FPM = 1.0

# FAR 91.159 asks a VFR cruise to be hemispheric only more than this above the
# ground. The rule is written in AGL; the engine has no terrain, so it is
# applied as MSL -- see `_check_vfr_hemispheric` for what that costs.
VFR_CRUISE_FLOOR_FT = 3000.0

SEVERITIES = ("error", "warning", "info")


@dataclass(frozen=True)
class Finding:
    """One thing about the plan that does not add up."""

    severity: str  # error | warning | info
    code: str  # stable slug, so the UI can style or suppress by kind
    message: str  # pilot-readable, says what and where
    row: int | None = None  # navlog row index, None for the plan as a whole

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")


@dataclass(frozen=True)
class ConsistencyReport:
    findings: tuple[Finding, ...]

    @property
    def is_consistent(self) -> bool:
        """True when nothing rises to an error.

        Warnings do not make a plan inconsistent -- they make it worth a second
        look. Only a contradiction counts.
        """
        return not any(f.severity == "error" for f in self.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")


@dataclass(frozen=True)
class _Context:
    """What every check gets: the log, and the assumptions behind it."""

    navlog: Navlog
    aircraft: Aircraft
    conditions: Conditions

    @property
    def flying(self) -> list[tuple[int, object]]:
        """Rows that are actually flown, with their navlog row index.

        Taxi and pattern rows burn time and fuel at a point; they have no
        course, no altitude change and no profile to be consistent with.
        """
        return [(index, leg) for index, leg in enumerate(self.navlog.legs) if leg.covers_ground]


def check_navlog_consistency(
    navlog: Navlog,
    *,
    aircraft: Aircraft | None = None,
    conditions: Conditions | None = None,
) -> ConsistencyReport:
    """Run every check over a finished navlog.

    Ordered most severe first, so the first thing the pilot reads is the thing
    most likely to matter.
    """
    from engine.navlog import Aircraft as _Aircraft
    from engine.navlog import Conditions as _Conditions

    ctx = _Context(
        navlog=navlog,
        aircraft=aircraft or _Aircraft(),
        conditions=conditions or _Conditions(),
    )
    findings: list[Finding] = []
    for check in _CHECKS:
        findings.extend(check(ctx))
    findings.sort(key=lambda f: (SEVERITIES.index(f.severity), f.row or -1))
    return ConsistencyReport(findings=tuple(findings))


# --- implemented checks --------------------------------------------------


def _check_altitude_continuity(ctx: _Context) -> list[Finding]:
    """Each leg must start where the one before it ended.

    A gap means the profile is claiming the aeroplane teleported, which is the
    signature of a leg whose altitude was overridden without the rest of the
    plan being re-flown.
    """
    findings: list[Finding] = []
    for (_, before), (index, after) in pairwise(ctx.flying):
        left = before.exit_altitude_ft
        right = after.entry_altitude_ft
        if left is None or right is None:
            continue
        if abs(left - right) > ALTITUDE_TOLERANCE_FT:
            findings.append(
                Finding(
                    severity="error",
                    code="altitude-discontinuity",
                    message=(
                        f"{before.to_name} is left at {left:.0f} ft but "
                        f"{after.from_name} is entered at {right:.0f} ft, a jump "
                        f"of {abs(left - right):.0f} ft with nothing flying it"
                    ),
                    row=index,
                )
            )
    return findings


def _check_declared_vs_actual(ctx: _Context) -> list[Finding]:
    """A leg must do what it says it does.

    Declaring "climb to" and then naming a lower altitude is the easiest
    mistake to make in user-driven mode, and the planner emits it as declared
    rather than second-guessing, so this is where it surfaces.
    """
    findings: list[Finding] = []
    for index, leg in ctx.flying:
        declared = leg.segment_type
        if declared is None:
            continue
        if leg.entry_altitude_ft is None or leg.exit_altitude_ft is None:
            continue
        change = leg.exit_altitude_ft - leg.entry_altitude_ft
        if abs(change) <= ALTITUDE_TOLERANCE_FT:
            actual = "cruise"
        else:
            actual = "climb" if change > 0 else "descent"
        if actual != declared:
            findings.append(
                Finding(
                    severity="error",
                    code="declared-vs-actual",
                    message=(
                        f"{leg.from_name} to {leg.to_name} is declared a "
                        f"{declared} but goes from {leg.entry_altitude_ft:.0f} ft "
                        f"to {leg.exit_altitude_ft:.0f} ft, which is a {actual}"
                    ),
                    row=index,
                )
            )
    return findings


def _check_arrival_altitude(ctx: _Context) -> list[Finding]:
    """The last leg of each flight must arrive at the field.

    Checked per flight, not just once, because every intermediate landing on a
    multi-stop day has to arrive somewhere too.
    """
    findings: list[Finding] = []
    flying = ctx.flying
    if not flying:
        return findings

    last_of_flight: dict[int, tuple[int, object]] = {}
    for index, leg in flying:
        last_of_flight[leg.flight_index] = (index, leg)

    for index, leg in last_of_flight.values():
        if leg.exit_altitude_ft is None:
            continue
        # The elevation is not on the row, so the pattern row that follows is
        # the only local witness to the field. Compare against sea level only
        # when there is nothing better -- an arrival still at cruise is worth
        # flagging regardless.
        if leg.exit_altitude_ft > 1.0 and leg.segment_type != "descent":
            findings.append(
                Finding(
                    severity="warning",
                    code="arrival-altitude",
                    message=(
                        f"the last leg into {leg.to_name} is a "
                        f"{leg.segment_type} ending at {leg.exit_altitude_ft:.0f} ft; "
                        f"an arrival is normally a descent to field elevation"
                    ),
                    row=index,
                )
            )
    return findings


def _check_stale_overrides(ctx: _Context) -> list[Finding]:
    """An override on a row that cannot use it did nothing.

    The ground rows never consult overrides, so an edit aimed at one is
    silently inert today. Silence is the problem: the pilot believes they
    changed something.
    """
    findings: list[Finding] = []
    for index, leg in enumerate(ctx.navlog.legs):
        if leg.overridden and not leg.covers_ground:
            findings.append(
                Finding(
                    severity="warning",
                    code="stale-override",
                    message=(
                        f"row {index + 1} ({leg.phase}) carries a manual edit, but "
                        f"a {leg.phase} row has no course, wind or airspeed to edit"
                    ),
                    row=index,
                )
            )
    return findings


def _check_climb_achievable(ctx: _Context) -> list[Finding]:
    """A climb must not demand more altitude than the leg has distance for.

    The planner emits what was declared even when the declaration is beyond the
    aeroplane -- an overridden altitude, or a manual-mode leg that simply asks
    for too much -- so the row reads as flyable and is silently flown with the
    time it would really take. `profile.reachable_altitude` bisects the POH
    climb table for the top actually attainable in the ground distance
    available, at this leg's own wind and temperature, and anything more than
    `CLIMB_TOLERANCE_FT` above that is a plan the aeroplane cannot fly.

    Only the over-reaching side is checked: topping out lower than the aeroplane
    could manage is a choice, not a contradiction.
    """
    from engine import profile

    findings: list[Finding] = []
    for index, leg in ctx.flying:
        if leg.segment_type != "climb":
            continue
        entry, exit_ = leg.entry_altitude_ft, leg.exit_altitude_ft
        if entry is None or exit_ is None or leg.distance_nm <= 0:
            continue
        geo = inverse(leg.from_position, leg.to_position)
        try:
            reachable = profile.reachable_altitude(
                entry, exit_, leg.distance_nm, geo, ctx.aircraft, ctx.conditions
            )
        except perf.OutsidePOHEnvelope:
            # A navlog that built cannot normally get here, but a check is the
            # last thing that should be the reason the report fails to appear.
            continue
        if exit_ > reachable + CLIMB_TOLERANCE_FT:
            findings.append(
                Finding(
                    severity="error",
                    code="climb-unreachable",
                    message=(
                        f"{leg.from_name} to {leg.to_name} climbs to "
                        f"{exit_:.0f} ft, but {leg.distance_nm:.1f} nm from "
                        f"{entry:.0f} ft reaches only {reachable:.0f} ft at the "
                        f"POH climb rate, {exit_ - reachable:.0f} ft short"
                    ),
                    row=index,
                )
            )
    return findings


def _check_descent_rate(ctx: _Context) -> list[Finding]:
    """A descent must not need to come down faster than the plan assumes.

    `Aircraft.descent_rate_fpm` is what the planner flies a descent at; a row
    whose arrival altitude was set by hand can imply something far steeper, and
    the row still shows a plausible ETE. What the row *implies* is the altitude
    it sheds over the time it takes, and that is what gets checked -- against
    `MAX_DESCENT_RATE_FPM` rather than the aircraft's own figure, so lowering
    the assumed rate cannot quietly lower the bar.
    """
    findings: list[Finding] = []
    for index, leg in ctx.flying:
        if leg.segment_type != "descent":
            continue
        entry, exit_ = leg.entry_altitude_ft, leg.exit_altitude_ft
        if entry is None or exit_ is None or leg.ete_min <= 0:
            continue
        rate = (entry - exit_) / leg.ete_min
        if rate > MAX_DESCENT_RATE_FPM + _DESCENT_RATE_EPSILON_FPM:
            findings.append(
                Finding(
                    severity="error",
                    code="descent-too-steep",
                    message=(
                        f"{leg.from_name} to {leg.to_name} sheds "
                        f"{entry - exit_:.0f} ft in {leg.ete_min:.1f} min, a "
                        f"descent of {rate:.0f} ft/min; the plan is flown at no "
                        f"more than {MAX_DESCENT_RATE_FPM:.0f} ft/min"
                    ),
                    row=index,
                )
            )
    return findings


def _legal_vfr_altitude(altitude_ft: float, *, wants_odd: bool) -> float:
    """The nearest thousand-plus-500 of the right parity to `altitude_ft`.

    An altitude of the wrong parity sits exactly between two legal ones, and
    the tie is broken downwards: the lower is the one already flown at least
    once on the way up.
    """
    thousands = round((altitude_ft - 500.0) / 1000.0)
    if thousands % 2 != (1 if wants_odd else 0):
        # Step to whichever neighbour of the right parity is the closer.
        below, above = thousands - 1, thousands + 1
        thousands = (
            below
            if abs(altitude_ft - (below * 1000.0 + 500.0))
            <= abs(altitude_ft - (above * 1000.0 + 500.0))
            else above
        )
    return thousands * 1000.0 + 500.0


def _check_vfr_hemispheric(ctx: _Context) -> list[Finding]:
    """A cruise above 3000 ft must sit on the hemispheric altitude for its course.

    FAR 91.159: magnetic course 0-179 flies odd thousands plus 500, 180-359
    even thousands plus 500. Only level flight is asked -- a climb or a descent
    passes through whatever lies between its ends.

    The floor is applied as **MSL**, not the AGL the rule is written in,
    because the engine has no terrain. Over rising ground that is permissive in
    the wrong direction: a leg at 3500 ft MSL over a 2000 ft ridge is under
    3000 AGL and outside the rule, and this still asks it to be hemispheric.
    Erring towards saying something is why the finding is a warning and not an
    error -- the plan is not contradicting itself, and it is the pilot's chart,
    not this module, that settles the AGL.
    """
    findings: list[Finding] = []
    for index, leg in ctx.flying:
        if leg.segment_type != "cruise":
            continue
        altitude = leg.exit_altitude_ft
        if altitude is None or altitude <= VFR_CRUISE_FLOOR_FT:
            continue
        course = leg.magnetic_course_deg % 360.0
        wants_odd = course < 180.0
        legal = _legal_vfr_altitude(altitude, wants_odd=wants_odd)
        if abs(altitude - legal) <= ALTITUDE_TOLERANCE_FT:
            continue
        findings.append(
            Finding(
                severity="warning",
                code="vfr-hemispheric",
                message=(
                    f"{leg.from_name} to {leg.to_name} cruises at "
                    f"{altitude:.0f} ft on a magnetic course of {course:.0f} deg, "
                    f"which asks for {'odd' if wants_odd else 'even'} thousands "
                    f"plus 500 under FAR 91.159; the nearest is {legal:.0f} ft"
                ),
                row=index,
            )
        )
    return findings


# --- stubs: the shape is settled, the rule is not ------------------------
#
# Each of these is a real check we intend to make. They are registered and
# tested as no-ops so that filling one in is a one-function change rather than
# a re-plumbing, and so the list of what is *not* yet checked is visible in the
# code rather than living in someone's head.


def _check_toc_before_tod(ctx: _Context) -> list[Finding]:
    """TODO: within one flight, a top of descent must not precede its top of climb.

    Rows carry `start_role` / `end_role`, so this is an ordering check over
    those, per `flight_index`.
    """
    return []


def _check_cruise_below_chart(ctx: _Context) -> list[Finding]:
    """TODO: flag a level leg below the cruise chart's floor.

    `navlog` already warns and reads the lowest published page. Repeating it
    here puts it in front of the pilot at go/no-go time, where the numbers are
    being trusted.
    """
    return []


_CHECKS: tuple[Callable[[_Context], list[Finding]], ...] = (
    _check_altitude_continuity,
    _check_declared_vs_actual,
    _check_arrival_altitude,
    _check_stale_overrides,
    _check_climb_achievable,
    _check_descent_rate,
    _check_toc_before_tod,
    _check_cruise_below_chart,
    _check_vfr_hemispheric,
)


def format_report(report: ConsistencyReport) -> str:
    """Render a report as text, for the copyable navlog."""
    if not report.findings:
        return "NAVLOG CONSISTENCY: no problems found."
    headline = (
        f"NAVLOG CONSISTENCY: {len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    )
    lines = [headline, "-" * 78]
    for finding in report.findings:
        where = f"row {finding.row + 1}" if finding.row is not None else "plan"
        lines.append(f"  [{finding.severity:<7}] {where:<8} {finding.message}")
    return "\n".join(lines)
