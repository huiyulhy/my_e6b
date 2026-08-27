"""The go/no-go checklist.

Preflight check should consist of:
1. Takeoff distance at departure airport < runway length with a margin
2. Landing distance at destination airport < runway length with a margin
3. Total fuel reserve is > 30 min for day VFR and > 45 min for night
4. Weight and balance within limits

Check is performed against all runways at the airports
"""

from __future__ import annotations

from dataclasses import dataclass

from engine import performance as perf

# Safety factor on runway and fuel reserves
DEFAULT_RUNWAY_MARGIN = 0.20
DEFAULT_FUEL_MARGIN = 0.10

# The dry-grass correction is applied to every unpaved surface. Gravel and
# dirt are not grass, and the POH publishes no figure for them, so the grass
# penalty is used as the nearest published number and the runway is flagged.
# Anything unrecognised is treated as paved
_UNPAVED_CODES = frozenset(
    {
        "turf", "grass", "gras", "grs", "sod",
        "dirt", "soil", "earth", "grvl", "gravel", "sand", "trtd",
    }
)
_PAVED_CODES = frozenset(
    {"asph", "asphalt", "asp", "conc", "concrete", "con", "paved", "pem", "bit"}
)


def _surface_parts(surface: str) -> list[str]:
    """Split a surface code into its parts: 'ASPH-TURF' -> ['asph', 'turf']."""
    return [
        part
        for part in surface.replace("/", "-").replace(" ", "-").lower().split("-")
        if part
    ]


@dataclass(frozen=True)
class Margins:
    """How much more than the book number the pilot wants before going.
    """

    runway: float = DEFAULT_RUNWAY_MARGIN
    fuel: float = DEFAULT_FUEL_MARGIN

    def __post_init__(self) -> None:
        if self.runway < 0:
            raise ValueError(f"runway margin {self.runway:g} cannot be negative")
        if self.fuel < 0:
            raise ValueError(f"fuel margin {self.fuel:g} cannot be negative")


@dataclass(frozen=True)
class Runway:
    """One runway at an airport, as the checklist needs it."""

    designation: str  # "12/30", or "" if unknown
    length_ft: float | None
    surface: str = ""
    lighted: bool = False

    @property
    def is_grass(self) -> bool:
        """Whether the POH's dry-grass correction applies."""
        return any(part in _UNPAVED_CODES for part in _surface_parts(self.surface))

    @property
    def surface_is_known(self) -> bool:
        parts = _surface_parts(self.surface)
        return bool(parts) and all(
            part in _UNPAVED_CODES or part in _PAVED_CODES for part in parts
        )


@dataclass(frozen=True)
class RunwayCheck:
    """One operation on one runway: book distance against length available."""

    airport: str
    operation: str  # takeoff | landing
    runway: str  # designation
    surface: str
    dry_grass_applied: bool

    runway_available_ft: float | None
    ground_roll_ft: float | None
    over_50ft_ft: float | None  # the book distance the check is made against
    required_ft: float | None  # over_50ft_ft with the margin applied

    passes: bool | None  # None when it could not be determined
    note: str = ""
    # True when the POH simply does not publish a number for these conditions.
    # That is a no-go rather than an unknown: the aeroplane is being asked to
    # operate outside the envelope the manufacturer tested it in.
    outside_envelope: bool = False

    @property
    def spare_ft(self) -> float | None:
        """Runway left over after the required distance. Negative fails."""
        if self.runway_available_ft is None or self.required_ft is None:
            return None
        return self.runway_available_ft - self.required_ft


@dataclass(frozen=True)
class AirportCheck:
    """Every runway at one airport, for one operation, and the verdict.

    The conditions the distances were computed at are carried here rather
    than on each runway, since they are the same for all of them and this is
    what the pilot wants to see beside the numbers.
    """

    airport: str
    operation: str  # takeoff | landing

    elevation_ft: float
    pressure_altitude_ft: float
    density_altitude_ft: float
    oat_c: float
    weight_lb: float
    margin: float

    runways: tuple[RunwayCheck, ...]

    @property
    def passes(self) -> bool | None:
        """True if any runway works, False if none does, None if unknown.

        None only when nothing could be determined at all -- if one runway is
        unmeasurable but another clearly works, the airport works.
        """
        if any(r.passes for r in self.runways):
            return True
        if any(r.passes is False for r in self.runways):
            return False
        return None

    @property
    def best(self) -> RunwayCheck | None:
        """The runway with the most room to spare, which is the one to use."""
        measured = [r for r in self.runways if r.spare_ft is not None]
        return max(measured, key=lambda r: r.spare_ft) if measured else None


@dataclass(frozen=True)
class FuelCheck:
    """Fuel at the end of the route against the reserve, plus margin."""

    fuel_on_board_gal: float
    burn_gal: float  # includes taxi
    landing_with_gal: float

    reserve_minutes: float  # 30 day, 45 night -- FAR 91.151
    reserve_required_gal: float  # the bare regulatory minimum
    required_with_margin_gal: float
    margin: float
    night: bool

    passes: bool

    @property
    def spare_gal(self) -> float:
        """Fuel above the margined reserve. Negative fails."""
        return self.landing_with_gal - self.required_with_margin_gal

    @property
    def spare_minutes(self) -> float | None:
        """The same slack expressed as time, which is how pilots think."""
        if self.reserve_required_gal <= 0 or self.reserve_minutes <= 0:
            return None
        gph = self.reserve_required_gal / (self.reserve_minutes / 60.0)
        return 60.0 * self.spare_gal / gph if gph > 0 else None


@dataclass(frozen=True)
class GoNoGo:
    """The whole checklist, and the single verdict that follows from it."""

    airports: tuple[AirportCheck, ...]
    fuel: FuelCheck
    blockers: tuple[str, ...]  # every reason this is a no-go
    unknowns: tuple[str, ...]  # checks that could not be made at all

    @property
    def is_go(self) -> bool:
        """True only when every check was made and every check passed."""
        return not self.blockers and not self.unknowns


def check_airport(
    *,
    airport: str,
    operation: str,
    runways: tuple[Runway, ...],
    elevation_ft: float,
    oat_c: float,
    pressure_altitude_ft: float,
    density_altitude_ft: float,
    weight_lb: float,
    margin: float,
) -> AirportCheck:
    """Book distances for one operation, against every runway on the field."""
    shared = {
        "airport": airport,
        "operation": operation,
        "elevation_ft": elevation_ft,
        "pressure_altitude_ft": pressure_altitude_ft,
        "density_altitude_ft": density_altitude_ft,
        "oat_c": oat_c,
        "weight_lb": weight_lb,
        "margin": margin,
    }

    if not runways:
        return AirportCheck(
            **shared,
            runways=(
                RunwayCheck(
                    airport=airport,
                    operation=operation,
                    runway="",
                    surface="",
                    dry_grass_applied=False,
                    runway_available_ft=None,
                    ground_roll_ft=None,
                    over_50ft_ft=None,
                    required_ft=None,
                    passes=None,
                    note="no runway data on file for this airport",
                ),
            ),
        )

    checks = [
        _check_one_runway(
            airport=airport,
            operation=operation,
            runway=runway,
            oat_c=oat_c,
            pressure_altitude_ft=pressure_altitude_ft,
            weight_lb=weight_lb,
            margin=margin,
        )
        for runway in runways
    ]
    return AirportCheck(**shared, runways=tuple(checks))


def _check_one_runway(
    *,
    airport: str,
    operation: str,
    runway: Runway,
    oat_c: float,
    pressure_altitude_ft: float,
    weight_lb: float,
    margin: float,
) -> RunwayCheck:
    dry_grass = runway.is_grass
    notes: list[str] = []
    if runway.surface and not runway.surface_is_known:
        notes.append(f"surface {runway.surface!r} not recognised; treated as paved")

    # A high altimeter setting at a low field puts the pressure altitude below
    # sea level, which is off the bottom of the POH chart. Read it at sea
    # level instead: the chart does not go lower, and the sea-level figure is
    # the longer of the two, so the error is in the safe direction.
    if pressure_altitude_ft < 0.0:
        notes.append(
            f"pressure altitude {pressure_altitude_ft:.0f} ft is below the "
            f"chart; read at sea level, which is conservative"
        )
        pressure_altitude_ft = 0.0

    try:
        if operation == "takeoff":
            distance = perf.takeoff_distance(
                weight_lb, pressure_altitude_ft, oat_c, dry_grass=dry_grass
            )
        else:
            distance = perf.landing_distance(
                pressure_altitude_ft, oat_c, dry_grass=dry_grass
            )
    except perf.OutsidePOHEnvelope as exc:
        return RunwayCheck(
            airport=airport,
            operation=operation,
            runway=runway.designation,
            surface=runway.surface,
            dry_grass_applied=dry_grass,
            runway_available_ft=runway.length_ft,
            ground_roll_ft=None,
            over_50ft_ft=None,
            required_ft=None,
            # A missing chart entry is a refusal, not a shrug: the conditions
            # are outside the tested envelope, so the operation is a no-go.
            passes=False,
            note="; ".join(
                [*notes, f"outside the published {operation} envelope: {exc}"]
            ),
            outside_envelope=True,
        )

    required = distance.total_over_50ft_ft * (1.0 + margin)
    if runway.length_ft is None:
        notes.append("no published length for this runway")

    return RunwayCheck(
        airport=airport,
        operation=operation,
        runway=runway.designation,
        surface=runway.surface,
        dry_grass_applied=dry_grass,
        runway_available_ft=runway.length_ft,
        ground_roll_ft=distance.ground_roll_ft,
        over_50ft_ft=distance.total_over_50ft_ft,
        required_ft=required,
        passes=None if runway.length_ft is None else runway.length_ft >= required,
        note="; ".join(notes),
    )


def check_fuel(
    *,
    fuel_on_board_gal: float,
    burn_gal: float,
    reserve_required_gal: float,
    reserve_minutes: float,
    margin: float,
    night: bool,
) -> FuelCheck:
    """Landing fuel against the FAR reserve with the pilot's margin on top."""
    landing_with = fuel_on_board_gal - burn_gal
    required = reserve_required_gal * (1.0 + margin)
    return FuelCheck(
        fuel_on_board_gal=fuel_on_board_gal,
        burn_gal=burn_gal,
        landing_with_gal=landing_with,
        reserve_minutes=reserve_minutes,
        reserve_required_gal=reserve_required_gal,
        required_with_margin_gal=required,
        margin=margin,
        night=night,
        passes=landing_with >= required,
    )


def summarise(airports: list[AirportCheck], fuel: FuelCheck) -> GoNoGo:
    """Collect the checks into a verdict with its reasons."""
    blockers: list[str] = []
    unknowns: list[str] = []

    for check in airports:
        if check.passes is False:
            longest = max(
                (r for r in check.runways if r.runway_available_ft is not None),
                key=lambda r: r.runway_available_ft,
                default=None,
            )
            needed = next(
                (r.required_ft for r in check.runways if r.required_ft is not None),
                None,
            )
            if needed is None:
                # No distance was computed at all, so there is nothing to
                # compare against the runway; the conditions themselves are
                # the blocker.
                reason = next(
                    (r.note for r in check.runways if r.outside_envelope and r.note),
                    "outside the published POH envelope",
                )
                blockers.append(f"{check.airport} {check.operation}: {reason}")
                continue
            blockers.append(
                f"{check.airport} {check.operation}: no runway is long enough -- "
                f"needs {needed:.0f} ft "
                f"(book plus {check.margin:.0%}) at a density altitude of "
                f"{check.density_altitude_ft:.0f} ft, longest available is "
                f"{longest.runway_available_ft:.0f} ft"
                + (f" ({longest.runway})" if longest and longest.runway else "")
            )
        elif check.passes is None:
            reason = next(
                (r.note for r in check.runways if r.note), "could not be checked"
            )
            unknowns.append(f"{check.airport} {check.operation}: {reason}")

    if not fuel.passes:
        blockers.append(
            f"fuel: lands with {fuel.landing_with_gal:.1f} gal but needs "
            f"{fuel.required_with_margin_gal:.1f} gal "
            f"({fuel.reserve_required_gal:.1f} gal "
            f"{fuel.reserve_minutes:.0f}-minute reserve plus {fuel.margin:.0%})"
        )

    return GoNoGo(
        airports=tuple(airports),
        fuel=fuel,
        blockers=tuple(blockers),
        unknowns=tuple(unknowns),
    )


def format_checklist(result: GoNoGo) -> str:
    """Render the checklist as text, for the copyable navlog."""
    lines = [
        f"GO / NO-GO: {'GO' if result.is_go else 'NO GO'}",
        "-" * 104,
    ]

    for check in result.airports:
        lines.append(
            f"{check.airport} {check.operation} -- field {check.elevation_ft:.0f} ft, "
            f"OAT {check.oat_c:.0f} C, pressure alt {check.pressure_altitude_ft:.0f} ft, "
            f"density alt {check.density_altitude_ft:.0f} ft, "
            f"{check.weight_lb:.0f} lb, margin {check.margin:.0%}"
        )
        lines.append(
            f"  {'RWY':<10}{'SURFACE':<12}{'LENGTH':>8}{'ROLL':>8}"
            f"{'BOOK50':>8}{'REQD':>8}{'SPARE':>8}  RESULT"
        )
        for runway in check.runways:
            if runway.outside_envelope:
                outcome = "NO DATA"
            elif runway.passes:
                outcome = "ok"
            else:
                outcome = "SHORT" if runway.passes is False else "?"
            lines.append(
                f"  {(runway.runway or '--'):<10}{(runway.surface or '--'):<12}"
                f"{_or_dash(runway.runway_available_ft, 8)}"
                f"{_or_dash(runway.ground_roll_ft, 8)}"
                f"{_or_dash(runway.over_50ft_ft, 8)}"
                f"{_or_dash(runway.required_ft, 8)}"
                f"{_or_dash(runway.spare_ft, 8)}  {outcome}"
            )
            if runway.note:
                lines.append(f"    {runway.note}")
        lines.append("")

    fuel = result.fuel
    lines.append(
        f"Fuel: {fuel.fuel_on_board_gal:.1f} gal on board, "
        f"{fuel.burn_gal:.1f} gal burnt, lands with {fuel.landing_with_gal:.1f} gal. "
        f"Needs {fuel.required_with_margin_gal:.1f} gal "
        f"({fuel.reserve_minutes:.0f}-minute "
        f"{'night' if fuel.night else 'day'} reserve of "
        f"{fuel.reserve_required_gal:.1f} gal plus {fuel.margin:.0%}). "
        f"{'ok' if fuel.passes else 'SHORT'}."
    )

    for blocker in result.blockers:
        lines.append(f"NO GO: {blocker}")
    for unknown in result.unknowns:
        lines.append(f"UNKNOWN: {unknown}")

    return "\n".join(lines)


def _or_dash(value: float | None, width: int) -> str:
    return f"{value:>{width}.0f}" if value is not None else f"{'--':>{width}}"
