"""The go/no-go checklist.

Preflight check should consist of:
1. Takeoff distance at departure airport < runway length with a margin
2. Landing distance at destination airport < runway length with a margin
3. Total fuel reserve is > 30 min for day VFR and > 45 min for night
4. Weight and balance within limits
5. The weather at the departure and landing fields is VFR

Check is performed against all runways at the airports

**The field has to be VFR, not merely legal.** A ceiling is checked against
the height the pattern is flown at rather than against the 1,000 ft in the
regulation: an overcast at 1,100 ft is legal to take off under and leaves
nowhere to fly a circuit, so an overcast has to sit at least 500 ft above the
pattern -- the same clearance 91.155 asks for below a cloud. A vertical
visibility is not a ceiling to fly under at all and ends the check. Weather is
only checked where there is weather to check: a field with no report gets the
verdict it got before, which is one about runways and nothing else.

**Wind is part of every distance.** A runway is not a length, it is a length
pointing somewhere: the same 2500 ft strip is comfortable into 12 kt and
marginal with 5 kt on the tail. So each runway is checked end by end -- the end
that gives the better headwind is the one the numbers are read for -- and the
crosswind on that end is checked against the maximum the type was demonstrated
in. A runway that is long enough but cannot be lined up with is not a runway
you can use.

Winds are handled in MAGNETIC degrees here, because that is what a runway
designator is. METAR and TAF winds are true; converting them is the caller's
job, since only the caller knows the field's magnetic variation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from engine import performance as perf

# Safety factor on runway and fuel reserves
DEFAULT_RUNWAY_MARGIN = 0.20
DEFAULT_FUEL_MARGIN = 0.10

# Basic VFR at the field: 3 statute miles and a 1,000 ft ceiling, which is
# what 14 CFR 91.155 asks for in the surface area of controlled airspace and
# what 91.157 asks for before special VFR is even on the table.
VFR_VISIBILITY_SM = 3.0
VFR_CEILING_FT_AGL = 1000.0

# How far below an overcast the pattern still has to fit. 500 ft below for VFR
PATTERN_CLEARANCE_FT = 500.0

DEFAULT_PATTERN_HEIGHT_AGL_FT = 1000.0

# Effectively IFR sky cover
OBSCURATION_COVERS = frozenset({"OVX", "VV"})

# The dry-grass correction is applied to every unpaved surface. 
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


# A runway designator: one or two digits of magnetic heading in tens, with an
# optional side letter for parallel runways. "09L", "9", "27R", "36".
_DESIGNATOR = re.compile(r"^(\d{1,2})([LRCWE]?)$")

@dataclass(frozen=True)
class RunwayEnd:
    """One direction of a runway, and the heading it points."""

    label: str  # "12", "27R"
    magnetic_heading_deg: float


@dataclass(frozen=True)
class SurfaceWind:
    """The wind over a field, referenced to MAGNETIC north.
    `gust_kt` is the peak when the report has one
    """

    from_deg: float
    speed_kt: float
    gust_kt: float | None = None

    def __post_init__(self) -> None:
        if self.speed_kt < 0:
            raise ValueError(f"wind speed {self.speed_kt:g} kt cannot be negative")
        if self.gust_kt is not None and self.gust_kt < self.speed_kt:
            raise ValueError(
                f"gust {self.gust_kt:g} kt is below the steady wind "
                f"{self.speed_kt:g} kt"
            )

    @property
    def peak_kt(self) -> float:
        return self.speed_kt if self.gust_kt is None else self.gust_kt


@dataclass(frozen=True)
class WindComponents:
    """The wind resolved onto one runway direction."""

    headwind_kt: float  # negative for a tailwind
    crosswind_kt: float  # magnitude
    crosswind_from_right: bool

    @property
    def is_tailwind(self) -> bool:
        return self.headwind_kt < 0


def wind_components(runway_heading_deg: float, wind: SurfaceWind) -> WindComponents:
    """Resolve a wind onto a runway heading, taking the unfavourable reading.
    Take the worst case scenario in gusts to calculate wind
    """
    off_rad = math.radians(wind.from_deg - runway_heading_deg)
    steady_head = wind.speed_kt * math.cos(off_rad)
    gust_head = wind.peak_kt * math.cos(off_rad)
    crosswind = wind.peak_kt * math.sin(off_rad)
    return WindComponents(
        headwind_kt=min(steady_head, gust_head),
        crosswind_kt=abs(crosswind),
        crosswind_from_right=crosswind >= 0,
    )


@dataclass(frozen=True)
class Runway:
    """One runway at an airport"""

    designation: str  # "12/30", or "" if unknown
    length_ft: float | None
    surface: str = ""
    lighted: bool = False

    @property
    def ends(self) -> tuple[RunwayEnd, ...]:
        """Both directions of the strip, with their magnetic headings.

        The heading comes from the designator rather than from a published
        alignment, so it is quantised to ten degrees
        """
        found: list[RunwayEnd] = []
        for part in self.designation.replace("-", "/").split("/"):
            match = _DESIGNATOR.match(part.strip().upper())
            if not match:
                continue
            tens = int(match.group(1))
            if not 1 <= tens <= 36:
                continue
            found.append(
                RunwayEnd(label=part.strip().upper(), magnetic_heading_deg=tens * 10.0)
            )
        return tuple(found)

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

    # The end the numbers were read for: the one with the better headwind.
    # Empty when there was no wind to choose with, or no heading to choose by.
    end_used: str = ""
    headwind_kt: float | None = None  # negative for a tailwind
    crosswind_kt: float | None = None
    crosswind_from_right: bool | None = None
    # A runway that cannot be lined up with is not usable, however long it is.
    crosswind_exceeds_demonstrated: bool = False

    runway_available_ft: float | None = None
    ground_roll_ft: float | None = None
    over_50ft_ft: float | None = None  # the book distance checked against
    required_ft: float | None = None  # over_50ft_ft with the margin applied

    # Cells read from somewhere other than where the query asked, straight
    # from the chart layer. Empty is the normal case: the whole reading came
    # off the published chart.
    off_chart: tuple[perf.OffChart, ...] = ()

    passes: bool | None = None  # None when it could not be determined
    note: str = ""
    # True when the POH simply does not publish a number for these conditions.
    # That is a no-go rather than an unknown: the aeroplane is being asked to
    # operate outside the envelope the manufacturer tested it in.
    outside_envelope: bool = False

    @property
    def extrapolated(self) -> bool:
        """Whether any number on this row came from off the chart."""
        return bool(self.off_chart)

    @property
    def optimistic(self) -> bool:
        """Whether any substitution here errs on the unsafe side."""
        return any(not entry.conservative for entry in self.off_chart)

    @property
    def spare_ft(self) -> float | None:
        """Runway left over after the required distance. Negative fails."""
        if self.runway_available_ft is None or self.required_ft is None:
            return None
        return self.runway_available_ft - self.required_ft


@dataclass(frozen=True)
class FieldWeather:
    """What a field's report says about flying VFR out of or into it.

    Built from a resolved `engine.weather.SurfaceWeather`, and deliberately
    only the parts the go/no-go decides on. `sky_reported` is the one that is
    easy to lose: a clear sky and a source that does not observe cloud both
    arrive with no ceiling in them, and they are opposite answers.
    """

    visibility_sm: float | None = None
    ceiling_ft_agl: float | None = None
    ceiling_cover: str = ""  # BKN | OVC | OVX | VV; "" when there is no ceiling
    sky_reported: bool = False

    @property
    def is_obscuration(self) -> bool:
        return self.ceiling_cover.upper() in OBSCURATION_COVERS


@dataclass(frozen=True)
class WeatherCheck:
    """The field weather against VFR, for one operation at one airport."""

    airport: str
    operation: str  # takeoff | landing

    visibility_sm: float | None
    ceiling_ft_agl: float | None
    ceiling_cover: str
    obscured: bool
    # Whether the report said anything about the sky at all -- the difference
    # between a clear sky and a source that does not observe cloud.
    sky_reported: bool

    # The pattern this field flies, and the ceiling that follows from it. The
    # requirement is carried rather than recomputed, because it depends on the
    # cover as well as the pattern and the pilot is owed the number that was
    # actually applied.
    pattern_altitude_agl_ft: float
    required_ceiling_ft_agl: float | None

    passes: bool | None  # None when there was not enough of a report to say
    # Every gate that closed, not just the first. A field can be below
    # minimums on visibility *and* under an overcast, and waiting out one of
    # them is not waiting out the other.
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        """The verdict in one line, for the checklist and the blocker list."""
        if self.reasons:
            return "; ".join(self.reasons)
        if self.passes is None:
            return "; ".join(self.notes) or "not enough weather to judge VFR"
        return "VFR"


def check_weather(
    *,
    airport: str,
    operation: str,
    weather: FieldWeather | None,
    pattern_altitude_agl_ft: float = DEFAULT_PATTERN_HEIGHT_AGL_FT,
) -> WeatherCheck | None:
    """The field against VFR minimums, plus room to fly the pattern.

    `None` in, `None` out: with no report there is nothing to check, and the
    checklist says nothing about the weather

    Any ceiling has to clear the 1,000 ft in the regulation (we need to fly 1000ft above); 
    an overcast has additionally to clear the pattern by 500 ft
    """
    if weather is None:
        return None

    cover = weather.ceiling_cover.upper()
    obscured = weather.is_obscuration
    reasons: list[str] = []
    notes: list[str] = []
    unknown = False

    required: float | None = None
    if obscured: # this means IFR
        height = (
            ""
            if weather.ceiling_ft_agl is None
            else f" at {weather.ceiling_ft_agl:.0f} ft"
        )
        reasons.append(
            f"sky obscured ({cover}{height}); this is not a ceiling to fly "
            f"VFR under"
        )
    elif weather.ceiling_ft_agl is not None:
        required = VFR_CEILING_FT_AGL
        if cover == "OVC":
            required = max(
                required, pattern_altitude_agl_ft + PATTERN_CLEARANCE_FT
            )
        if weather.ceiling_ft_agl < required:
            reasons.append(
                f"{cover or 'ceiling'} {weather.ceiling_ft_agl:.0f} ft AGL is "
                f"below the {required:.0f} ft needed"
                + (
                    f" to fly the {pattern_altitude_agl_ft:.0f} ft pattern "
                    f"{PATTERN_CLEARANCE_FT:.0f} ft below an overcast"
                    if cover == "OVC"
                    and required > VFR_CEILING_FT_AGL
                    else " for VFR"
                )
            )
    elif not weather.sky_reported:
        # No cloud group at all. Common on a model-only forecast, which
        # observes no sky -- and a forecast that cannot see cloud is not a
        # forecast of no cloud.
        unknown = True
        notes.append("no sky condition in this report; the ceiling is unknown")

    if weather.visibility_sm is None:
        notes.append("no visibility in this report; it was not checked")
        unknown = True
    elif weather.visibility_sm < VFR_VISIBILITY_SM:
        reasons.append(
            f"visibility {weather.visibility_sm:g} sm is below the "
            f"{VFR_VISIBILITY_SM:g} sm VFR minimum"
        )

    return WeatherCheck(
        airport=airport,
        operation=operation,
        visibility_sm=weather.visibility_sm,
        ceiling_ft_agl=weather.ceiling_ft_agl,
        ceiling_cover=cover,
        obscured=obscured,
        sky_reported=weather.sky_reported,
        pattern_altitude_agl_ft=pattern_altitude_agl_ft,
        required_ceiling_ft_agl=required,
        passes=False if reasons else (None if unknown else True),
        reasons=tuple(reasons),
        notes=tuple(notes),
    )


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

    # The wind the distances were read in, magnetic. None means none was
    # given, and the runways carry the no-wind book figures.
    wind: SurfaceWind | None

    runways: tuple[RunwayCheck, ...]

    # The field's weather against VFR. `None` where no report was given, which
    # is not the same as a report that could not be judged -- see
    # `check_weather`.
    weather: WeatherCheck | None = None

    @property
    def runways_pass(self) -> bool | None:
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
    def passes(self) -> bool | None:
        """The field as a whole: the runways and the weather over them.

        Both are gates, and on the same footing as length and crosswind are on
        a single runway. The longest runway on the field is no use under an
        obscuration, and a clear sky does not lengthen a short one.
        """
        verdicts = [self.runways_pass]
        if self.weather is not None:
            verdicts.append(self.weather.passes)
        if any(v is False for v in verdicts):
            return False
        if any(v is None for v in verdicts):
            return None
        return True

    @property
    def off_chart(self) -> tuple[perf.OffChart, ...]:
        """Every off-chart reading behind this field's distances, deduped.

        Deduped because the conditions are the field's, not the runway's: a
        pressure altitude below the bottom row produces the identical record on
        every runway on the field, and listing it once per runway would bury
        the one that differs.
        """
        seen: dict[tuple[str, str], perf.OffChart] = {}
        for runway in self.runways:
            for entry in runway.off_chart:
                seen.setdefault((entry.what, entry.detail), entry)
        return tuple(seen.values())

    @property
    def extrapolated(self) -> bool:
        return bool(self.off_chart)

    @property
    def best(self) -> RunwayCheck | None:
        """The runway to use: one that works, with the most room to spare.

        A runway that fails outright is ranked below every runway that does
        not, so a long strip with an unflyable crosswind never comes back as
        the recommendation ahead of a shorter one into wind.
        """
        measured = [r for r in self.runways if r.spare_ft is not None]
        if not measured:
            return None
        return max(measured, key=lambda r: (r.passes is not False, r.spare_ft))


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
class Extrapolation:
    """One off-chart reading, and where in the plan it happened.

    `perf.OffChart` says what was read instead; this adds where. Kept
    structured rather than flattened to a sentence because the renderers need
    to sort and colour by `conservative`, and re-deriving that from prose is
    the kind of thing that works until the prose changes.
    """

    where: str  # "KSQL takeoff", "leg 3 (TOC to TOD)"
    what: str
    detail: str
    conservative: bool

    @property
    def summary(self) -> str:
        """The one-line form, for the text checklist."""
        tail = "" if self.conservative else " -- this one can read optimistic"
        return f"{self.where}: {self.detail}{tail}"


@dataclass(frozen=True)
class GoNoGo:
    """The whole checklist, and the single verdict that follows from it."""

    airports: tuple[AirportCheck, ...]
    fuel: FuelCheck
    blockers: tuple[str, ...]  # every reason this is a no-go
    unknowns: tuple[str, ...]  # checks that could not be made at all
    # Readings that came from somewhere other than where the query asked --
    # see `perf.OffChart`. Conservative ones first, so the ones that matter
    # are the ones the list finishes on.
    extrapolations: tuple[Extrapolation, ...] = ()

    @property
    def is_go(self) -> bool:
        """True only when every check was made and every check passed.

        Off-chart readings do not enter this. They are not failures -- every
        one of them is a published cell standing in for a query beside it --
        and letting them decide would put "NO GO" on an ordinary high-pressure
        morning at a sea-level field, which is both wrong and the fastest way
        to teach a pilot to ignore the word. They change `verdict` instead.
        """
        return not self.blockers and not self.unknowns

    @property
    def optimistic_extrapolations(self) -> tuple[Extrapolation, ...]:
        """The subset whose substitution errs on the unsafe side."""
        return tuple(e for e in self.extrapolations if not e.conservative)

    @property
    def all_from_the_book(self) -> bool:
        """True when every number behind this verdict is a published one."""
        return not self.extrapolations

    @property
    def verdict(self) -> str:
        """The single line at the top of the checklist.

        Three states, not two. "GO" means every number came off the chart as
        printed; "GO -- EXTRAPOLATED" means the answer is yes but some of it
        was read from air the POH does not publish for this operating point,
        and the reader is owed the list before they act on it.
        """
        if not self.is_go:
            return "NO GO"
        return "GO" if self.all_from_the_book else "GO -- EXTRAPOLATED"


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
    wind: SurfaceWind | None = None,
    weather: FieldWeather | None = None,
    pattern_altitude_agl_ft: float = DEFAULT_PATTERN_HEIGHT_AGL_FT,
) -> AirportCheck:
    """Book distances for one operation, against every runway on the field.

    `wind` is magnetic, and optional: with none the runways carry the no-wind
    book figures, which is what the charts publish and what the checklist did
    before wind was modelled. Every runway then says so in its note, because a
    distance read without a wind is a distance read at an assumption.

    `weather` is optional on the same terms, and omitted leaves the field's
    VFR check out of the verdict entirely rather than failing it.
    """
    shared = {
        "airport": airport,
        "operation": operation,
        "elevation_ft": elevation_ft,
        "pressure_altitude_ft": pressure_altitude_ft,
        "density_altitude_ft": density_altitude_ft,
        "oat_c": oat_c,
        "weight_lb": weight_lb,
        "margin": margin,
        "wind": wind,
        "weather": check_weather(
            airport=airport,
            operation=operation,
            weather=weather,
            pattern_altitude_agl_ft=pattern_altitude_agl_ft,
        ),
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
            wind=wind,
        )
        for runway in runways
    ]
    return AirportCheck(**shared, runways=tuple(checks))


def _pick_end(
    runway: Runway, wind: SurfaceWind | None
) -> tuple[RunwayEnd | None, WindComponents | None]:
    """The end to use, and the wind on it.

    The end with the greater headwind, which is the one a pilot picks and the
    only one the distances should be read for. Both ends see the same
    crosswind, so the choice is never between a crosswind and a headwind.
    """
    ends = runway.ends
    if wind is None or not ends:
        return None, None
    scored = [(end, wind_components(end.magnetic_heading_deg, wind)) for end in ends]
    return max(scored, key=lambda pair: pair[1].headwind_kt)


def _check_one_runway(
    *,
    airport: str,
    operation: str,
    runway: Runway,
    oat_c: float,
    pressure_altitude_ft: float,
    weight_lb: float,
    margin: float,
    wind: SurfaceWind | None = None,
) -> RunwayCheck:
    dry_grass = runway.is_grass
    notes: list[str] = []
    if runway.surface and not runway.surface_is_known:
        notes.append(f"surface {runway.surface!r} not recognised; treated as paved")

    end, components = _pick_end(runway, wind)
    headwind_kt = 0.0 if components is None else components.headwind_kt
    if wind is None:
        notes.append("no surface wind given; distances are the no-wind book figures")
    elif components is None:
        notes.append(
            f"runway {runway.designation!r} gives no heading, so the wind could "
            f"not be resolved onto it; distances are the no-wind book figures"
        )
    wind_facts = {
        "end_used": "" if end is None else end.label,
        "headwind_kt": None if components is None else components.headwind_kt,
        "crosswind_kt": None if components is None else components.crosswind_kt,
        "crosswind_from_right": (
            None if components is None else components.crosswind_from_right
        ),
        "crosswind_exceeds_demonstrated": (
            components is not None
            and components.crosswind_kt > perf.MAX_DEMONSTRATED_CROSSWIND_KT
        ),
    }
    if wind_facts["crosswind_exceeds_demonstrated"]:
        notes.append(
            f"crosswind {components.crosswind_kt:.0f} kt from the "
            f"{'right' if components.crosswind_from_right else 'left'} exceeds the "
            f"{perf.MAX_DEMONSTRATED_CROSSWIND_KT:.0f} kt maximum demonstrated"
        )

    # A high altimeter setting at a low field puts the pressure altitude below
    # the bottom of the POH chart, and a strong wind puts the headwind past the
    # end of the correction. Both used to be clamped here; both are now the
    # chart layer's business, which reads the nearest published cell and hands
    # back an `OffChart` saying it did. One place decides what the chart can
    # answer, and nothing downstream has to guess whether a number is the
    # book's.
    try:
        if operation == "takeoff":
            distance = perf.takeoff_distance(
                weight_lb,
                pressure_altitude_ft,
                oat_c,
                headwind_kt=headwind_kt,
                dry_grass=dry_grass,
            )
        else:
            distance = perf.landing_distance(
                pressure_altitude_ft,
                oat_c,
                headwind_kt=headwind_kt,
                dry_grass=dry_grass,
            )
    except perf.OutsidePOHEnvelope as exc:
        return RunwayCheck(
            airport=airport,
            operation=operation,
            runway=runway.designation,
            surface=runway.surface,
            dry_grass_applied=dry_grass,
            **wind_facts,
            runway_available_ft=runway.length_ft,
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

    long_enough = None if runway.length_ft is None else runway.length_ft >= required
    return RunwayCheck(
        airport=airport,
        operation=operation,
        runway=runway.designation,
        surface=runway.surface,
        dry_grass_applied=dry_grass,
        **wind_facts,
        runway_available_ft=runway.length_ft,
        ground_roll_ft=distance.ground_roll_ft,
        over_50ft_ft=distance.total_over_50ft_ft,
        required_ft=required,
        off_chart=distance.off_chart,
        # Length and crosswind are both gates: passing one does not excuse the
        # other, and the crosswind verdict is known even where the length is
        # not, so it decides an otherwise-unknown runway.
        passes=False if wind_facts["crosswind_exceeds_demonstrated"] else long_enough,
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


def summarise(
    airports: list[AirportCheck],
    fuel: FuelCheck,
    *,
    off_chart: Iterable[tuple[str, perf.OffChart]] = (),
) -> GoNoGo:
    """Collect the checks into a verdict with its reasons.

    `off_chart` carries readings from outside this module -- the cruise chart
    substitutions the navlog hits on a hot day, each paired with the leg it
    happened on. The airports' own off-chart readings are collected from the
    checks themselves and do not need passing in.
    """
    blockers: list[str] = []
    unknowns: list[str] = []

    for check in airports:
        where = f"{check.airport} {check.operation}"
        # Weather and runways are reported separately even though either one
        # sinks the field. "No runway is long enough" and "the field is under
        # an overcast" are different problems with different answers, and a
        # pilot who fixes one by taking less fuel still needs to know about
        # the other.
        if check.weather is not None:
            if check.weather.passes is False:
                blockers.append(f"{where} weather: {check.weather.summary}")
            elif check.weather.passes is None:
                unknowns.append(f"{where} weather: {check.weather.summary}")

        if check.runways_pass is False:
            blockers.append(f"{where}: {_why_no_runway(check)}")
        elif check.runways_pass is None:
            reason = next(
                (r.note for r in check.runways if r.note), "could not be checked"
            )
            unknowns.append(f"{where}: {reason}")

    if not fuel.passes:
        blockers.append(
            f"fuel: lands with {fuel.landing_with_gal:.1f} gal but needs "
            f"{fuel.required_with_margin_gal:.1f} gal "
            f"({fuel.reserve_required_gal:.1f} gal "
            f"{fuel.reserve_minutes:.0f}-minute reserve plus {fuel.margin:.0%})"
        )

    # Conservative readings first, optimistic last: the list is read top down
    # and the one worth stopping on should be the one it ends on.
    found: list[tuple[str, perf.OffChart]] = [
        (f"{check.airport} {check.operation}", entry)
        for check in airports
        for entry in check.off_chart
    ]
    found.extend(off_chart)
    found.sort(key=lambda pair: pair[1].conservative, reverse=True)
    extrapolations = tuple(
        Extrapolation(
            where=where,
            what=entry.what,
            detail=entry.detail,
            conservative=entry.conservative,
        )
        for where, entry in found
    )

    return GoNoGo(
        airports=tuple(airports),
        fuel=fuel,
        blockers=tuple(blockers),
        unknowns=tuple(unknowns),
        extrapolations=extrapolations,
    )


def _why_no_runway(check: AirportCheck) -> str:
    """Why no runway on this field works, in the terms that actually decided it.
    """
    crosswind_out = [r for r in check.runways if r.crosswind_exceeds_demonstrated]
    no_data = [r for r in check.runways if r.outside_envelope]
    # The runways that got as far as having a distance to compare: the only
    # ones a length verdict can be about.
    measured = [
        r
        for r in check.runways
        if r.required_ft is not None and not r.crosswind_exceeds_demonstrated
    ]

    if not measured:
        reasons: list[str] = []
        if crosswind_out:
            least = min(r.crosswind_kt for r in crosswind_out)
            reasons.append(
                f"{_plural(len(crosswind_out), 'runway')} "
                f"{_is_are(len(crosswind_out))} above the "
                f"{perf.MAX_DEMONSTRATED_CROSSWIND_KT:.0f} kt maximum demonstrated "
                f"crosswind (the best is {least:.0f} kt)"
            )
        seen: set[str] = set()
        for runway in no_data:
            note = runway.note or "outside the published POH envelope"
            if note not in seen:
                seen.add(note)
                reasons.append(note)
        if not reasons:
            return "no runway could be measured"
        wind = "" if check.wind is None else f"wind {_format_wind(check.wind)} leaves "
        return f"{wind}no usable runway -- " + "; ".join(reasons)

    # The nearest miss on length, which is the runway the pilot would have tried.
    with_length = [r for r in measured if r.runway_available_ft is not None]
    closest = max(with_length, key=lambda r: r.spare_ft) if with_length else measured[0]
    named = f" ({closest.runway}{'/' + closest.end_used if closest.end_used else ''})"
    have = (
        f"{closest.runway_available_ft:.0f} ft available"
        if closest.runway_available_ft is not None
        else "no published length"
    )
    reason = (
        f"no runway is long enough -- the best{named} needs "
        f"{closest.required_ft:.0f} ft (book plus {check.margin:.0%}) at a density "
        f"altitude of {check.density_altitude_ft:.0f} ft against {have}"
    )
    if crosswind_out:
        reason += (
            f", and {_plural(len(crosswind_out), 'runway')} "
            f"{_is_are(len(crosswind_out))} out on crosswind"
        )
    return reason


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _is_are(count: int) -> str:
    return "is" if count == 1 else "are"


def format_checklist(result: GoNoGo) -> str:
    """Render the checklist as text, for the copyable navlog."""
    lines = [
        f"GO / NO-GO: {result.verdict}",
        "-" * 123,
    ]

    for check in result.airports:
        lines.append(
            f"{check.airport} {check.operation} -- field {check.elevation_ft:.0f} ft, "
            f"OAT {check.oat_c:.0f} C, pressure alt {check.pressure_altitude_ft:.0f} ft, "
            f"density alt {check.density_altitude_ft:.0f} ft, "
            f"{check.weight_lb:.0f} lb, margin {check.margin:.0%}, "
            f"wind {_format_wind(check.wind)}"
        )
        if check.weather is not None:
            lines.append(f"  Weather: {_format_weather(check.weather)}")
        lines.append(
            f"  {'RWY':<10}{'SURFACE':<12}{'USE':>5}{'HEAD':>7}{'XWIND':>7}"
            f"{'LENGTH':>8}{'ROLL':>8}"
            f"{'BOOK50':>8}{'REQD':>8}{'SPARE':>8}  RESULT"
        )
        for runway in check.runways:
            if runway.outside_envelope:
                outcome = "NO DATA"
            elif runway.crosswind_exceeds_demonstrated:
                outcome = "XWIND"
            elif runway.passes:
                outcome = "ok"
            else:
                outcome = "SHORT" if runway.passes is False else "?"
            lines.append(
                f"  {(runway.runway or '--'):<10}{(runway.surface or '--'):<12}"
                f"{(runway.end_used or '--'):>5}"
                f"{_or_dash(runway.headwind_kt, 7)}"
                f"{_or_dash(runway.crosswind_kt, 7)}"
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
    # Last, under the verdict they qualify. These are not failures, so they do
    # not belong among the blockers; they are the reason the word above them
    # may say GO with a qualifier on it.
    for entry in result.extrapolations:
        lines.append(f"EXTRAPOLATED: {entry.summary}")

    return "\n".join(lines)


def _format_weather(check: WeatherCheck) -> str:
    """The sky and the visibility, then what the checklist made of them."""
    if check.obscured:
        sky = f"sky obscured ({check.ceiling_cover})"
    elif check.ceiling_ft_agl is not None:
        sky = f"{check.ceiling_cover or 'ceiling'} {check.ceiling_ft_agl:.0f} ft AGL"
    elif check.sky_reported:
        sky = "no ceiling"
    else:
        sky = "sky not reported"
    visibility = (
        "visibility not reported"
        if check.visibility_sm is None
        else f"{check.visibility_sm:g} sm visibility"
    )
    verdict = "VFR" if check.passes else ("NOT VFR" if check.passes is False else "?")
    detail = "" if check.passes else f" -- {check.summary}"
    return (
        f"{sky}, {visibility}, pattern {check.pattern_altitude_agl_ft:.0f} ft AGL. "
        f"{verdict}{detail}"
    )


def _or_dash(value: float | None, width: int) -> str:
    return f"{value:>{width}.0f}" if value is not None else f"{'--':>{width}}"


def _format_wind(wind: SurfaceWind | None) -> str:
    """A wind in the shape a pilot reads it, magnetic: '270M at 12G18 kt'."""
    if wind is None:
        return "not given"
    if wind.speed_kt == 0:
        return "calm"
    gust = "" if wind.gust_kt is None else f"G{wind.gust_kt:.0f}"
    return f"{wind.from_deg:03.0f}M at {wind.speed_kt:.0f}{gust} kt"
