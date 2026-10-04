"""Cessna 172S POH Section 5 performance tables and interpolation:
- Takeoff distance charts
- Landing distance chart
- Climb distance charts
- Cruise performance charts

Main rules:
1. Never extrapolate beyond table boundaries
2. Interpolate linearly between published nodes, except on the pressure
   altitude axis of the takeoff and landing distance charts (log linear interp)

Table structure, which is not uniform and drives the code below:
* takeoff  -- weight x pressure altitude x temperature, but weight is read as
  a chart selector rather than an interpolation axis: a query is rounded up to
  the next published weight and then interpolated in 2-D on pressure altitude
  and temperature
* landing  -- 2-D grid: pressure altitude x temperature, at 2550 lb only
* climb rate -- 2-D grid: pressure altitude x temperature
* climb time/fuel -- 1-D cumulative-from-sea-level table at standard
  temperature, read as the difference between two rows; the same table's
  climb speed column is read as a plain 1-D lookup
* cruise   -- RAGGED: which RPM settings exist depends on altitude (2100 RPM
  only appears at 2000-4000 ft, 2700 RPM only at 8000-10000 ft), so this one
  cannot use a regular grid. It is stored one altitude page at a time, and
  unlike every other chart here its altitude is a *selector rather than an
  interpolation axis*: the query is rounded down to the page at or below it,
  which is then read at temperature and then RPM. Rounding down is the
  conservative direction on fuel and it is what makes the ragged chart
  readable at all. See `cruise`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from engine.atmosphere import (
    density_altitude,
    isa_temperature_c,
    oat_for_density_altitude,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "poh" / "c172s"

# Temperatures encoded in the takeoff/landing column names, in Celsius.
_WIDE_TEMPS_C = (0.0, 10.0, 20.0, 30.0, 40.0)

# Airframe constants for the 172S. Verify against POH Section 1/2.
MAX_GROSS_WEIGHT_LB = 2550.0
FUEL_CAPACITY_GAL = 53.0
FUEL_USABLE_GAL = 50.0
BEST_GLIDE_KIAS = 68.0
GLIDE_RATIO_NM_PER_1000FT = 1.5

# Maximum demonstrated crosswind, POH Section 1. Not a certificated limit --
# it is the strongest crosswind the type was flight tested in -- but it is the
# only number the manufacturer publishes, so the checklist treats exceeding it
# as a no-go rather than as advice.
MAX_DEMONSTRATED_CROSSWIND_KT = 15.0

# The takeoff and landing charts publish their wind correction for headwinds
# up to 30 kt and tailwinds up to 10 kt. Past the tailwind end the correction
# is an extrapolation of a penalty that is already steep (10% per 2 kt), so the
# query is refused instead.
MAX_CHART_TAILWIND_KT = 10.0
MAX_CHART_HEADWIND_KT = 30.0

# How far outside the cruise chart's published temperature columns a reading
# may be taken before it counts as an extrapolation.
CRUISE_ISA_TOLERANCE_C = 0.0

# Taxi takeoff fuel consumption (based on POH)
START_TAXI_TAKEOFF_FUEL_GAL = 1.4

class OutsidePOHEnvelope(ValueError):
    """A query fell outside the range of published POH data."""

# --- result types --------------------------------------------------------


@dataclass(frozen=True)
class OffChart:
    """ Which reading was not published in the POH
    
    `conservative` is the field that decides what a caller should do about it.
    True means the substitute errs on the safe side -- it reads a longer
    ground roll, or a lower rate of climb, than the aeroplane will actually
    deliver -- and a plan built on it is still a plan you can fly. False means
    the substitute errs the other way, or is an approximation with error in
    both directions, and the number is optimistic or merely close: that is the
    kind that has to reach the pilot before the go/no-go is read as a "GO".
    """

    what: str  # short label, e.g. "pressure altitude"
    detail: str  # a sentence a pilot can act on
    conservative: bool


@dataclass(frozen=True)
class GroundDistance:
    """A takeoff or landing distance pair, in feet."""

    ground_roll_ft: float
    total_over_50ft_ft: float

    # Every cell that was read from somewhere other than where the query
    # asked. Empty is the normal case and means the whole reading is straight
    # off the published chart.
    off_chart: tuple[OffChart, ...] = ()

    @property
    def extrapolated(self) -> bool:
        """Whether any part of this distance came from off the chart."""
        return bool(self.off_chart)

    @property
    def optimistic(self) -> bool:
        """Whether any substitution errs on the unsafe side."""
        return any(not entry.conservative for entry in self.off_chart)


@dataclass(frozen=True)
class ClimbRate:
    fpm: float
    kias: float


@dataclass(frozen=True)
class ClimbSegment:
    """Time, fuel and airspeed for a climb between two pressure altitudes.

    No distance: the published climb distance is a no-wind, standard-day
    figure, so the caller flies `kias` against the forecast wind instead --
    see `profile._solve_change`.
    """

    time_min: float
    fuel_gal: float
    kias: float


@dataclass(frozen=True)
class CruisePoint:
    percent_power: float
    ktas: float
    gph: float


# --- CSV loading ---------------------------------------------------------

def _read_csv(name: str, data_dir: Path | None = None) -> list[dict[str, float]]:
    """Read one POH CSV into a list of float-valued dicts.

    Blank cells become NaN rather than an error - to be used for data integrity checks
    """
    path = (data_dir or DATA_DIR) / f"{name}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            {k: (float(v) if v and v.strip() else float("nan")) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def _grid_axes(rows: list[dict[str, float]], *keys: str) -> tuple[np.ndarray, ...]:
    """Sorted distinct values for each named column, i.e. the grid axes."""
    return tuple(np.array(sorted({row[key] for row in rows})) for key in keys)


def _melt_wide_temps(
    rows: list[dict[str, float]], prefix: str, *index_keys: str
) -> dict[tuple[float, ...], list[float]]:
    """Pull the temperature-suffixed columns into a value list per index key.
    """
    out: dict[tuple[float, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[k] for k in index_keys)
        out[key] = [row[f"{prefix}_{int(t)}"] for t in _WIDE_TEMPS_C]
    return out

# --- interpolator construction -------------------------------------------
@dataclass(frozen=True)
class _Grid:
    """A bounds-checked interpolator that tolerates chart holes.
    Every axis blends linearly except, optionally, the one named by
    `log_axis`
    `values` keeps NaN wherever the chart is blank; 
    """

    axes: tuple[np.ndarray, ...]
    values: np.ndarray
    published: np.ndarray
    log_axis: int | None


def _regular(
    axes: tuple[np.ndarray, ...],
    values: np.ndarray,
    *,
    log_axis: int | None = None,
) -> _Grid:
    published = ~np.isnan(values)
    if log_axis is not None and not np.all(values[published] > 0.0):
        raise ValueError(
            "log-space interpolation needs strictly positive published values"
        )
    # Blank cells are filled with 1.0 rather than 0.0 so the arithmetic stays
    # finite in log space
    return _Grid(
        axes=axes,
        values=np.where(published, values, 1.0),
        published=published.astype(float),
        log_axis=log_axis,
    )


def _bracket(axis: np.ndarray, x: float) -> tuple[slice, float]:
    """Locate `x` on one axis, as the pair of nodes around it and a fraction.

    Raises IndexError outside the axis; the caller turns that into the
    envelope error, since we never extrapolate.
    """
    if not (axis[0] - 1e-9 <= x <= axis[-1] + 1e-9):
        raise IndexError
    if len(axis) == 1:
        return slice(0, 1), 0.0
    k = min(max(int(np.searchsorted(axis, x, side="right")) - 1, 0), len(axis) - 2)
    return slice(k, k + 2), float((x - axis[k]) / (axis[k + 1] - axis[k]))


def _blend(block: np.ndarray, pos: int, frac: float, geometric: bool) -> np.ndarray:
    """Collapse one axis of the surrounding hypercube down to the query point."""
    lo = np.take(block, 0, axis=pos)
    hi = np.take(block, 1, axis=pos) if block.shape[pos] == 2 else lo
    if geometric:
        return np.exp(np.log(lo) * (1.0 - frac) + np.log(hi) * frac)
    return lo + frac * (hi - lo)


def _contract(block: np.ndarray, fracs: list[float], log_axis: int | None) -> float:
    """Reduce the hypercube one axis at a time down to a single value.

    Linear axes go first and the log axis last, so the geometric blend acts on
    values already resolved at the queried temperature -- the way the chart is
    read, picking the temperature column before working down the altitude
    rows. Contracting in the other order would give a slightly different
    answer, since geometric and arithmetic blends do not commute.
    """
    order = [i for i in range(len(fracs)) if i != log_axis]
    if log_axis is not None:
        order.append(log_axis)
    remaining = list(range(len(fracs)))
    for axis in order:
        pos = remaining.index(axis)
        block = _blend(block, pos, fracs[axis], geometric=axis == log_axis)
        remaining.pop(pos)
    return float(block)


def _call(grid: _Grid, point: tuple[float, ...], what: str) -> float:
    """Evaluate a grid, converting any refusal into our own error.

    Two ways a query can fail: it falls outside the grid entirely, or it draws
    on a blank chart cell, which the validity mask reports. Both mean the POH
    does not publish an answer.
    """
    try:
        brackets = [_bracket(axis, x) for axis, x in zip(grid.axes, point)]
    except IndexError as exc:
        bounds = ", ".join(f"[{ax.min():g}, {ax.max():g}]" for ax in grid.axes)
        raise OutsidePOHEnvelope(
            f"{what} query {point} is outside the published POH range {bounds}"
        ) from exc
    window = tuple(s for s, _ in brackets)
    fracs = [f for _, f in brackets]
    # The mask is always contracted linearly: it measures how much of the
    # query's support lands on published cells, which is a linear question.
    if _contract(grid.published[window], fracs, None) < 1.0 - 1e-9:
        raise OutsidePOHEnvelope(
            f"the POH publishes no {what} at {point}; the chart is blank there, "
            f"which puts the operation outside the tested envelope"
        )
    return _contract(grid.values[window], fracs, grid.log_axis)


@dataclass(frozen=True)
class _CruiseTable:
    """The cruise chart, stored one altitude page at a time.

    The chart is ragged: each altitude publishes its own list of RPM settings,
    and the list slides upward and narrows with height. So rather than force
    it into a grid, each altitude keeps its own RPM axis and its own block of
    values -- which is how the POH prints it, one page per altitude.

    `values[i]` has shape (rpm, ISA deviation, 3), the last axis holding
    percent power, KTAS and GPH together.

    The third axis is an **ISA deviation, not an absolute temperature**. The
    POH prints the cruise table's columns as "20 C BELOW STANDARD / STANDARD /
    20 C ABOVE STANDARD", so one column means the same thing at every altitude
    even though the absolute temperature it stands for falls with height.
    """

    altitudes: np.ndarray
    isa_devs: np.ndarray
    rpms: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]

    def rpm_range(self, index: int) -> tuple[float, float]:
        axis = self.rpms[index]
        return float(axis[0]), float(axis[-1])


@dataclass(frozen=True)
class _Tables:
    """All interpolators, built once and reused."""

    takeoff_roll: _Grid
    takeoff_50: _Grid
    takeoff_liftoff_kias: _Grid
    takeoff_50_kias: _Grid
    landing_roll: _Grid
    landing_50: _Grid
    climb_rate: _Grid
    climb_kias: _Grid
    climb_cum_time: _Grid
    climb_cum_fuel: _Grid
    climb_table_kias: _Grid
    cruise: _CruiseTable

@lru_cache(maxsize=4)
def _tables(data_dir: Path | None = None) -> _Tables:
    return _build_tables(data_dir)


def _build_tables(data_dir: Path | None = None) -> _Tables:
    # -- takeoff: weight x pressure altitude x temperature ----------------
    rows = _read_csv("takeoff", data_dir)
    wts, palts = _grid_axes(rows, "wt", "p_alt")
    temps = np.array(_WIDE_TEMPS_C)
    roll = _melt_wide_temps(rows, "groundroll", "wt", "p_alt")
    over50 = _melt_wide_temps(rows, "clearfifty", "wt", "p_alt")
    shape = (len(wts), len(palts), len(temps))
    roll_v = np.empty(shape)
    over_v = np.empty(shape)
    for i, w in enumerate(wts):
        for j, p in enumerate(palts):
            roll_v[i, j, :] = roll[(w, p)]
            over_v[i, j, :] = over50[(w, p)]
    # Liftoff and 50 ft speeds vary with weight only.
    speed_by_wt = {row["wt"]: (row["liftoff"], row["speed50"]) for row in rows}
    lift_v = np.array([speed_by_wt[w][0] for w in wts])
    s50_v = np.array([speed_by_wt[w][1] for w in wts])

    # -- landing: pressure altitude x temperature, 2550 lb only -----------
    rows = _read_csv("landing", data_dir)
    (l_palts,) = _grid_axes(rows, "p_alt")
    l_roll = _melt_wide_temps(rows, "groundroll", "p_alt")
    l_over = _melt_wide_temps(rows, "clearfifty", "p_alt")
    l_roll_v = np.array([l_roll[(p,)] for p in l_palts])
    l_over_v = np.array([l_over[(p,)] for p in l_palts])

    # -- climb rate: pressure altitude x temperature ----------------------
    rows = _read_csv("max_climb_rate", data_dir)
    (c_palts,) = _grid_axes(rows, "p_alt")
    c_temps = np.array([-20.0, 0.0, 20.0, 40.0])
    by_palt = {row["p_alt"]: row for row in rows}
    rate_v = np.array(
        [[by_palt[p][f"t_{int(t)}"] for t in c_temps] for p in c_palts]
    )
    kias_v = np.array([by_palt[p]["kias"] for p in c_palts])

    # -- climb cumulative time/fuel and climb speed, 1-D at standard temp --
    rows = _read_csv("climb_dist", data_dir)
    rows.sort(key=lambda r: r["p_alt"])
    d_palts = np.array([r["p_alt"] for r in rows])
    cum_time = np.array([r["cum_time"] for r in rows])
    cum_fuel = np.array([r["cum_fuel"] for r in rows])
    table_kias = np.array([r["speed"] for r in rows])

    # -- cruise: one page per altitude, each with its own RPM axis --------
    rows = _read_csv("cruise", data_dir)
    cr_devs = np.array(sorted({r["isa_dev_c"] for r in rows}))
    cr_alts = np.array(sorted({r["press_alt"] for r in rows}))
    by_point = {(r["press_alt"], r["rpm"], r["isa_dev_c"]): r for r in rows}
    cr_rpms: list[np.ndarray] = []
    cr_values: list[np.ndarray] = []
    for alt in cr_alts:
        axis = np.array(sorted({r["rpm"] for r in rows if r["press_alt"] == alt}))
        page = np.full((len(axis), len(cr_devs), 3), np.nan)
        for i, rpm in enumerate(axis):
            for j, dev in enumerate(cr_devs):
                row = by_point.get((alt, rpm, dev))
                if row is not None:
                    page[i, j] = (row["pwr"], row["ktas"], row["gph"])
        if np.isnan(page).any():
            # Every published (altitude, RPM) is printed at every deviation
            # column. If that ever stops being true the sequential scheme below
            # needs a validity check, so fail loudly rather than drift.
            raise ValueError(
                f"cruise page at {alt:g} ft is missing an ISA deviation column"
            )
        cr_rpms.append(axis)
        cr_values.append(page)

    return _Tables(
        takeoff_roll=_regular((wts, palts, temps), roll_v, log_axis=1),
        takeoff_50=_regular((wts, palts, temps), over_v, log_axis=1),
        takeoff_liftoff_kias=_regular((wts,), lift_v),
        takeoff_50_kias=_regular((wts,), s50_v),
        landing_roll=_regular((l_palts, temps), l_roll_v, log_axis=0),
        landing_50=_regular((l_palts, temps), l_over_v, log_axis=0),
        climb_rate=_regular((c_palts, c_temps), rate_v),
        climb_kias=_regular((c_palts,), kias_v),
        climb_cum_time=_regular((d_palts,), cum_time),
        climb_cum_fuel=_regular((d_palts,), cum_fuel),
        climb_table_kias=_regular((d_palts,), table_kias),
        cruise=_CruiseTable(
            altitudes=cr_alts,
            isa_devs=cr_devs,
            rpms=tuple(cr_rpms),
            values=tuple(cr_values),
        ),
    )


# --- public performance queries ------------------------------------------
def _chart_pressure_altitude(pressure_altitude_ft: float) -> float:
    """Pressure altitude as the charts can read it, floored at sea level.

    Every chart with a pressure altitude axis starts at 0, but pressure
    altitude itself does not: a high altimeter setting puts a low field below
    the bottom row. At Palo Alto (4 ft) anything above 29.93 inHg does it,
    which is an ordinary high-pressure morning rather than an edge case, and
    refusing there fails the whole plan over air the POH simply does not print.

    Safe in the one direction that matters. Below 0 ft pressure altitude the
    air is *denser* than the bottom row, so the aeroplane beats what it reads
    there -- shorter ground rolls, a better rate of climb, a shorter climb.
    Reading sea level understates all of it, which is the conservative side, so
    unlike the cruise chart's 2000 ft floor this needs no warning: that one
    clamps in the direction that flatters the fuel flow, and says so.

    Conservative, but still not what was asked for, so it comes back with an
    `OffChart` saying so rather than passing for a published reading.
    """
    if pressure_altitude_ft >= 0.0:
        return pressure_altitude_ft, None
    return 0.0, OffChart(
        what="pressure altitude",
        detail=(
            f"a pressure altitude of {pressure_altitude_ft:.0f} ft is below the "
            f"bottom row of the chart; read at sea level, where the air is "
            f"thinner than it really is, so the distance errs long"
        ),
        conservative=True,
    )


def _select_weight(axis: np.ndarray, weight_lb: float | None) -> float:
    """Pick the published weight chart for takeoff, rounding a query upward.

    The POH prints a separate takeoff chart per weight, and a pilot reads the
    next chart at or above their actual weight rather than interpolating
    between two of them -- that keeps the answer on the conservative side.
    Above the heaviest published chart there is nothing to round up to, so the
    query is refused. Omitting the weight reads the gross-weight chart.
    """
    if weight_lb is None:
        weight_lb = MAX_GROSS_WEIGHT_LB
    heavier = axis[axis >= weight_lb - 1e-9]
    if not len(heavier):
        raise OutsidePOHEnvelope(
            f"takeoff weight {weight_lb:g} lb is above the heaviest published "
            f"chart ({axis.max():g} lb)"
        )
    return float(heavier[0])


def takeoff_distance(
    weight_lb: float | None = None,
    pressure_altitude_ft: float | None = None,
    oat_c: float | None = None,
    *,
    headwind_kt: float = 0.0,
    dry_grass: bool = False,
    data_dir: Path | None = None,
) -> GroundDistance:
    """Short-field takeoff distance, flaps 10, from POH Section 5.

    `weight_lb` selects which published weight chart to read: it is rounded up
    to the next published weight, defaulting to maximum gross when omitted.
    Pressure altitude and temperature are then interpolated within that chart.
    """
    if pressure_altitude_ft is None or oat_c is None:
        raise TypeError("pressure_altitude_ft and oat_c are required")
    t = _tables(data_dir)
    chart_weight = _select_weight(t.takeoff_roll.axes[0], weight_lb)
    chart_alt, floored = _chart_pressure_altitude(pressure_altitude_ft)
    point = (chart_weight, chart_alt, oat_c)
    roll = _call(t.takeoff_roll, point, "takeoff ground roll")
    over = _call(t.takeoff_50, point, "takeoff distance over 50 ft")
    roll, over, capped = _apply_wind(roll, over, headwind_kt)
    if dry_grass:
        # The correction is a percentage of the ground roll, and it applies to
        # the over-50 ft figure as the same number of feet.
        penalty = 0.15 * roll
        roll += penalty
        over += penalty
    return GroundDistance(
        roll, over, off_chart=tuple(e for e in (floored, capped) if e is not None)
    )


def landing_distance(
    pressure_altitude_ft: float,
    oat_c: float,
    *,
    headwind_kt: float = 0.0,
    dry_grass: bool = False,
    data_dir: Path | None = None,
) -> GroundDistance:
    """Short-field landing distance, flaps 30, at 2550 lb.
    """
    t = _tables(data_dir)
    chart_alt, floored = _chart_pressure_altitude(pressure_altitude_ft)
    point = (chart_alt, oat_c)
    roll = _call(t.landing_roll, point, "landing ground roll")
    over = _call(t.landing_50, point, "landing distance over 50 ft")
    roll, over, capped = _apply_wind(roll, over, headwind_kt)
    if dry_grass:
        penalty = 0.45 * roll
        roll += penalty
        over += penalty
    return GroundDistance(
        roll, over, off_chart=tuple(e for e in (floored, capped) if e is not None)
    )


def _apply_wind(
    roll: float, over: float, headwind_kt: float
) -> tuple[float, float, OffChart | None]:
    """Apply POH wind corrections, shared by the takeoff and landing charts. (based on 172s)
    1. Add 15% of takeoff for dry grass
    2. Subtract 10% per 9 kts of headwind, add 10% per 2 kts of tailwind

    The correction is published over a limited band. Beyond the tailwind end it
    is refused: 10% per 2 kt compounds fast, and a number extrapolated off the
    end of that slope is not one to commit a takeoff to. A headwind past the
    published end is credited only as far as the chart goes, which errs long.
    """
    if headwind_kt < -MAX_CHART_TAILWIND_KT:
        raise OutsidePOHEnvelope(
            f"tailwind of {-headwind_kt:.0f} kt is beyond the "
            f"{MAX_CHART_TAILWIND_KT:.0f} kt the chart corrects for"
        )
    capped: OffChart | None = None
    if headwind_kt > MAX_CHART_HEADWIND_KT:
        capped = OffChart(
            what="headwind",
            detail=(
                f"a headwind of {headwind_kt:.0f} kt is past the "
                f"{MAX_CHART_HEADWIND_KT:.0f} kt the chart corrects for; credited "
                f"at {MAX_CHART_HEADWIND_KT:.0f} kt, so the distance errs long"
            ),
            conservative=True,
        )
        headwind_kt = MAX_CHART_HEADWIND_KT
    if headwind_kt >= 0:
        factor = 1.0 - 0.10 * (headwind_kt / 9.0)
    else:
        factor = 1.0 + 0.10 * (-headwind_kt / 2.0)
    factor = max(factor, 0.0)
    return roll * factor, over * factor, capped


def climb_rate(
    pressure_altitude_ft: float, oat_c: float, *, data_dir: Path | None = None
) -> ClimbRate:
    """Maximum rate of climb at gross weight, from POH Section 5."""
    t = _tables(data_dir)
    # The floor's own `OffChart` is dropped here: below sea level the aeroplane
    # out-climbs the bottom row, so the reading is conservative in the only
    # direction a climb matters, and `ClimbRate` has no caller that carries a
    # record. Takeoff, landing and cruise, which do, keep theirs.
    chart_alt, _ = _chart_pressure_altitude(pressure_altitude_ft)
    fpm = _call(t.climb_rate, (chart_alt, oat_c), "climb rate")
    kias = _call(t.climb_kias, (chart_alt,), "climb speed")
    return ClimbRate(fpm, kias)


def climb_from_to(
    from_pressure_altitude_ft: float,
    to_pressure_altitude_ft: float,
    *,
    oat_c: float | None = None,
    data_dir: Path | None = None,
) -> ClimbSegment:
    """Time, fuel and climb speed between two pressure altitudes.

    Time and fuel are the differences of the cumulative columns, each
    interpolated at the two altitudes: a climb to 6,500 ft is charged half way
    between the 6,000 and 7,000 ft rows, not rounded out to 7,000. Rounding
    out was tried and dropped -- on top of a table already printed at max
    gross and full throttle, it planned climbs minutes longer than they flew,
    and pushed the top of climb, and so the whole cruise, down the route.

    The climb speed is the average of the table's speed column at the two ends,
    which is the speed that represents the segment as a whol

    POH:
    1. Published table is at standard temperature
    2. Increase climb time, fuel and distance by 10% for each 10 degC
    """
    # Checked before the floor is applied, so a "climb" that really descends is
    # still a caller error rather than two clamped altitudes quietly agreeing.
    if to_pressure_altitude_ft < from_pressure_altitude_ft:
        raise ValueError("climb segment must end above where it starts")
    bottom, _ = _chart_pressure_altitude(from_pressure_altitude_ft)
    top, _ = _chart_pressure_altitude(to_pressure_altitude_ft)
    t = _tables(data_dir)
    kias = 0.5 * (
        _call(t.climb_table_kias, (bottom,), "climb speed")
        + _call(t.climb_table_kias, (top,), "climb speed")
    )
    time_min = _call(t.climb_cum_time, (top,), "climb time") - _call(
        t.climb_cum_time, (bottom,), "climb time"
    )
    fuel_gal = _call(t.climb_cum_fuel, (top,), "climb fuel") - _call(
        t.climb_cum_fuel, (bottom,), "climb fuel"
    )
    if oat_c is not None:
        # The note is read against the midpoint of the climb, the altitude that
        # represents the segment as a whole. Floored well above zero: no real
        # forecast is 50 degC below standard, and a climb that takes no time at
        # all is worse than a stale correction.
        mid = 0.5 * (bottom + top)
        deviation_c = oat_c - isa_temperature_c(mid)
        factor = max(0.5, 1.0 + 0.10 * (deviation_c / 10.0))
        time_min *= factor
        fuel_gal *= factor
    return ClimbSegment(time_min=time_min, fuel_gal=fuel_gal, kias=kias)


def cruise(
    pressure_altitude_ft: float,
    rpm: float,
    oat_c: float,
    *,
    data_dir: Path | None = None,
) -> CruisePoint:
    """Cruise power, true airspeed and fuel flow at a given power setting.

    The altitude is a **chart selector, not an interpolation axis**: it is
    rounded *down* to the published page at or below the query, and that one
    page is then read at temperature and then RPM. Two reasons, in order of
    weight:

    1. Conservative on fuel. A lower page is denser air and more power, so it
       reads a higher fuel burn per nautical mile than the true altitude
       would -- 0.0789 against 0.0769 gal/nm at 7,900 ft read on the 6,000 ft
       page at 2500 RPM, about 2.6%, which is the worst case on this chart.
    2. It is what makes the ragged chart readable. RPM settings drop off the
       top of the chart as it climbs -- 2100 and 2550 above 4,000 ft, 2200
       above 8,000 ft -- so blending two pages let the *upper* page's omission
       refuse a setting the pilot's own altitude publishes perfectly well.
       Reading one page asks only whether that page has the setting.

    The only refusal left is a setting missing from the page at or below the
    query, which happens where the setting is not available that low yet:
    2650 below 6,000 ft, 2700 below 8,000 ft. That is a real limit on the
    power available up there, not a gap to interpolate across.

    The cost is that TAS is read optimistically for the same reason fuel flow
    is read conservatively. That 6,000 ft page reads 114 KTAS where 7,900 ft
    would give 112.1, so a leg plans about a minute per 100 nm quicker than it
    will fly. Fuel per nautical mile still errs the safe way, but ETAs and
    leg times taken off this chart run slightly early, and anything downstream
    reading ground speed inherits that.

    Assumes the POH cruise condition: 2550 lb, recommended lean mixture.
    """
    table = _tables(data_dir).cruise

    isa_dev = oat_c - isa_temperature_c(pressure_altitude_ft)
    devs = table.isa_devs
    if not (devs[0] <= isa_dev <= devs[-1]):
        raise OutsidePOHEnvelope(
            f"cruise temperature {oat_c:g} C at {pressure_altitude_ft:g} ft is "
            f"ISA{isa_dev:+.0f}, outside the published "
            f"ISA{devs[0]:+.0f} to ISA{devs[-1]:+.0f} range"
        )

    alts = table.altitudes
    if not (alts[0] - 1e-9 <= pressure_altitude_ft <= alts[-1] + 1e-9):
        raise OutsidePOHEnvelope(
            f"cruise pressure altitude {pressure_altitude_ft:g} ft is outside "
            f"the published range [{alts[0]:g}, {alts[-1]:g}]"
        )
    # Round down to the published page at or below the query. The epsilon
    # keeps a query sitting exactly on a page on that page rather than on the
    # one below it.
    index = int(np.searchsorted(alts, pressure_altitude_ft + 1e-9, side="right")) - 1
    page = min(max(index, 0), len(alts) - 1)

    values = _cruise_page(table, page, rpm, isa_dev)
    return CruisePoint(float(values[0]), float(values[1]), float(values[2]))


def _cruise_page(
    table: _CruiseTable, page: int, rpm: float, isa_dev_c: float
) -> np.ndarray:
    """Read one altitude page at a deviation column and an RPM, in that order.

    `page` has already been chosen by `cruise` as the published altitude at or
    below the query, so this never blends pages and the RPM axis it consults
    is the one axis that matters.
    """
    devs, axis, values = table.isa_devs, table.rpms[page], table.values[page]
    altitude = table.altitudes[page]

    # 1. Temperature, collapsing the page to one value per published RPM.
    j = min(
        max(int(np.searchsorted(devs, isa_dev_c, side="right")) - 1, 0), len(devs) - 2
    )
    t_frac = (isa_dev_c - devs[j]) / (devs[j + 1] - devs[j])
    by_rpm = values[:, j, :] + t_frac * (values[:, j + 1, :] - values[:, j, :])

    # 2. RPM, within this page's own published settings. Outside them the POH
    #    is not silent by accident: since the page is the one at or below the
    #    query, a setting missing from it is one that is not available this
    #    low -- a limit on the power there is, not a hole in the chart.
    lo, hi = table.rpm_range(page)
    if not (lo - 1e-9 <= rpm <= hi + 1e-9):
        raise OutsidePOHEnvelope(
            f"the POH publishes no {rpm:g} RPM cruise setting on the {altitude:g} ft "
            f"page, which runs {lo:g} to {hi:g} RPM. That is the page at or below "
            f"the query, so the setting is not available that low."
        )
    i = min(max(int(np.searchsorted(axis, rpm, side="right")) - 1, 0), len(axis) - 2)
    r_frac = (rpm - axis[i]) / (axis[i + 1] - axis[i])
    return by_rpm[i] + r_frac * (by_rpm[i + 1] - by_rpm[i])


# --- reading the cruise chart at an equal density -------------------------
#
# The cruise chart publishes three temperature columns per altitude page:
# ISA-20, ISA and ISA+20. A hot summer afternoon at a low cruise altitude
# leaves that band easily -- ISA+25 over California in July is unremarkable --
# and refusing the whole plan for it is the wrong answer to a routine day.
#
# What saves it is that a normally aspirated engine and the airframe in front
# of it both care about *density*, not about pressure altitude and temperature
# separately. The same air occurs at many (pressure altitude, temperature)
# pairs, and some of those pairs are published. So an operating point that is
# too hot for its own page is read at another page describing the same air.
#
# This is a SUBSTITUTION, NOT AN EXTRAPOLATION. Every number still comes from
# inside the published grid; what moves is which cell is consulted, and it
# moves along a line of constant density, which is the variable the chart is
# really keyed to. It is still an approximation the POH does not bless -- the
# equivalence ignores the small part of cruise performance that follows
# temperature rather than density (fuel metering, exhaust-back-pressure and
# ram effects) -- so every caller is told when one has happened rather than
# being handed a number that looks published.
#
# How big that part is, measured across the chart: reading the ISA+20 column
# of a page against the standard-temperature page at the same density altitude
# agrees to within about 1.6% on both fuel flow and true airspeed, in either
# direction. That is the size of the approximation, and it is also the size of
# the step at the seam -- a plan does not move smoothly from the last
# published column to the first substituted reading, it jumps by up to that
# much. Small against a POH chart digitized to the nearest knot and tenth of a
# gallon, but real, and the reason the substitution is a fallback rather than
# the normal path.
#
# It was about 1% while `cruise` interpolated its altitude axis. Rounding down
# to a page moves the substituted reading by up to a page of its own, and the
# two disagreements compound.
#
# The chart's total reach in density altitude is about -480 ft to 14,270 ft.
# Beyond that there is no equal-density page to move to, and the refusal
# stands.


@dataclass(frozen=True)
class CruiseLookup:
    """A cruise chart reading, and where on the chart it was actually read.

    `substituted` is the flag that matters: False means the pilot's own
    pressure altitude and temperature were in the published band and this is
    simply the chart. True means the point was outside the temperature band
    and was read at an equal density elsewhere on the chart, with
    `pressure_altitude_ft` and `oat_c` saying where.
    """

    point: CruisePoint
    density_altitude_ft: float
    # Where the chart was read. Equal to the query when `substituted` is False.
    pressure_altitude_ft: float
    oat_c: float
    substituted: bool
    # What was actually asked for. Carried so the reading can describe itself
    # without the caller having to hold on to the query to make sense of it --
    # and because "ISA+27 was read at ISA+0" is the whole story, and only half
    # of it lives in the fields above.
    asked_pressure_altitude_ft: float = 0.0
    asked_oat_c: float = 0.0

    @property
    def asked_isa_deviation_c(self) -> float:
        """How far off standard the query itself was."""
        return self.asked_oat_c - isa_temperature_c(self.asked_pressure_altitude_ft)

    @property
    def isa_deviation_c(self) -> float:
        """How far off standard the page it was read at is."""
        return self.oat_c - isa_temperature_c(self.pressure_altitude_ft)

    @property
    def extrapolated(self) -> bool:
        """Whether this reading is an extrapolation of the cruise chart.

        The chart prints ISA-20, ISA and ISA+20 at each altitude. A query
        inside that band is read off the page it belongs to. Outside it there
        is no column for this operating point, and what came back is the same
        *density* found elsewhere on the chart -- a different pressure
        altitude and a different temperature, chosen because the air matches.

        That is an extrapolation of the chart in the sense that matters: the
        POH does not publish this aeroplane's cruise performance at this
        pressure altitude and this temperature, and the number standing in for
        it was measured somewhere else. `CRUISE_ISA_TOLERANCE_C` sets how far
        past the printed columns that judgement waits.
        """
        if not self.substituted:
            return False
        low, high = cruise_extrapolation_band()
        return not low - 1e-9 <= self.asked_isa_deviation_c <= high + 1e-9

    @property
    def off_chart(self) -> tuple[OffChart, ...]:
        """The extrapolation, in the vocabulary the rest of the plan uses.

        Marked `conservative=False`. Not because the equal-density reading is
        careless -- every number still comes from inside the published grid --
        but because its error runs both ways: about 1.6% on fuel flow and true
        airspeed, in either direction, with a step of up to that much at the
        seam. An approximation that can read a plan *better* than it will fly
        is one the pilot is entitled to see before treating the answer as the
        book's.

        Empty for a substitution inside the tolerance: the reading was taken
        beside the query, but close enough to the printed column that calling
        it an extrapolation would cry wolf.
        """
        if not self.extrapolated:
            return ()
        return (
            OffChart(
                what="cruise temperature",
                detail=(
                    f"the POH publishes no cruise performance at "
                    f"{self.asked_pressure_altitude_ft:.0f} ft and "
                    f"ISA{self.asked_isa_deviation_c:+.0f} -- the chart's "
                    f"columns run ISA{cruise_isa_band()[0]:+.0f} to "
                    f"ISA{cruise_isa_band()[1]:+.0f}. Extrapolated from "
                    f"{self.pressure_altitude_ft:.0f} ft and "
                    f"ISA{self.isa_deviation_c:+.0f}, the nearest published air "
                    f"of the same density ({self.density_altitude_ft:.0f} ft), "
                    f"which agrees to about 1.6% on fuel flow and true airspeed"
                ),
                conservative=False,
            ),
        )


def cruise_density_range(
    pressure_altitude_ft: float, *, data_dir: Path | None = None
) -> tuple[float, float]:
    """The density altitudes one pressure altitude is published across.

    The chart's coldest and hottest columns -- ISA-20 and ISA+20 -- turned
    into the density altitudes they represent at this pressure altitude. A
    query whose density altitude falls inside this pair can be answered on
    this page; one that does not needs another page or nothing at all.
    """
    devs = _tables(data_dir).cruise.isa_devs
    isa = isa_temperature_c(pressure_altitude_ft)
    return (
        density_altitude(pressure_altitude_ft, isa + float(devs[0])),
        density_altitude(pressure_altitude_ft, isa + float(devs[-1])),
    )


def cruise_density_pages(
    data_dir: Path | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Every published cruise altitude with the density band it covers.

    One `(pressure_altitude_ft, density_altitude_lo_ft, density_altitude_hi_ft)`
    per page, lowest first. The bands overlap heavily -- each spans about 4,750
    ft of density altitude while the pages are 2,000 ft apart -- so their union
    is continuous, which is why the search in `cruise_at_density` can always
    land somewhere if the density altitude is in range at all.
    """
    return tuple(
        (float(alt), *cruise_density_range(float(alt), data_dir=data_dir))
        for alt in _tables(data_dir).cruise.altitudes
    )


def cruise_at_density(
    pressure_altitude_ft: float,
    rpm: float,
    oat_c: float,
    *,
    data_dir: Path | None = None,
) -> CruiseLookup:
    """`cruise`, falling back to an equal-density reading when it has to.

    Three steps, in order:

    1. **The page the query belongs on.** If the temperature is inside the
       published ISA-20 to ISA+20 band, this is an ordinary lookup and nothing
       below happens.

    2. **The same density at standard temperature.** At ISA the density
       altitude *is* the pressure altitude, so a query whose density altitude
       falls inside the chart's altitude range is read there, at standard
       temperature. This is the common case for a hot day: ISA+25 at 6,500 ft
       is read as standard air at about 9,400 ft.

    3. **The nearest page that covers this density.** Where step 2 lands off
       the top or bottom of the altitude range there is no standard-temperature
       page to use, so the pages are searched from the highest down for the
       first whose density band contains the query, and the temperature that
       puts *this* air on *that* page is solved for exactly. A density altitude
       of 13,500 ft, off the top of the 12,000 ft chart, is read at 12,000 ft
       and ISA+13.

    Raises `OutsidePOHEnvelope` when no page covers the density at all, and
    lets the RPM refusals from `cruise` through untouched -- a setting the POH
    omits at the substituted altitude is still a setting that should not be
    used there.
    """
    table = _tables(data_dir).cruise
    devs = table.isa_devs
    isa_dev = oat_c - isa_temperature_c(pressure_altitude_ft)
    density_alt = density_altitude(pressure_altitude_ft, oat_c)

    # 1. In band: the chart answers for itself.
    if devs[0] <= isa_dev <= devs[-1]:
        return CruiseLookup(
            point=cruise(pressure_altitude_ft, rpm, oat_c, data_dir=data_dir),
            density_altitude_ft=density_alt,
            pressure_altitude_ft=pressure_altitude_ft,
            oat_c=oat_c,
            substituted=False,
            asked_pressure_altitude_ft=pressure_altitude_ft,
            asked_oat_c=oat_c,
        )

    alts = table.altitudes
    # 2. The same air, at standard temperature, where the chart has a page for
    #    it. Handed over as a raw density altitude rather than snapped to a
    #    page here: `cruise` rounds it down to one like any other query, so the
    #    substitution stays conservative the same way the normal path is.
    if alts[0] <= density_alt <= alts[-1]:
        equivalent_oat = isa_temperature_c(density_alt)
        return CruiseLookup(
            point=cruise(density_alt, rpm, equivalent_oat, data_dir=data_dir),
            density_altitude_ft=density_alt,
            pressure_altitude_ft=density_alt,
            oat_c=equivalent_oat,
            substituted=True,
            asked_pressure_altitude_ft=pressure_altitude_ft,
            asked_oat_c=oat_c,
        )

    # 3. Off the top or the bottom: the highest page whose band still covers
    #    this air, read at the temperature that puts the air there.
    for page_alt, low, high in reversed(cruise_density_pages(data_dir=data_dir)):
        if low <= density_alt <= high:
            # Clamped into the published columns. The page was chosen because
            # its band covers this air, so the clamp only ever absorbs the
            # rounding in the round trip through density -- without it a query
            # landing exactly on a band edge comes back a fraction of a
            # picodegree outside it and is refused for nothing.
            page_isa = isa_temperature_c(page_alt)
            equivalent_oat = min(
                max(
                    oat_for_density_altitude(page_alt, density_alt),
                    page_isa + float(devs[0]),
                ),
                page_isa + float(devs[-1]),
            )
            return CruiseLookup(
                point=cruise(page_alt, rpm, equivalent_oat, data_dir=data_dir),
                density_altitude_ft=density_alt,
                pressure_altitude_ft=page_alt,
                oat_c=equivalent_oat,
                substituted=True,
                asked_pressure_altitude_ft=pressure_altitude_ft,
                asked_oat_c=oat_c,
            )

    pages = cruise_density_pages(data_dir=data_dir)
    raise OutsidePOHEnvelope(
        f"cruise temperature {oat_c:g} C at {pressure_altitude_ft:g} ft is "
        f"ISA{isa_dev:+.0f}, outside the published ISA{devs[0]:+.0f} to "
        f"ISA{devs[-1]:+.0f} range, and its density altitude of "
        f"{density_alt:.0f} ft is outside the {min(p[1] for p in pages):.0f} to "
        f"{max(p[2] for p in pages):.0f} ft the chart covers at any page"
    )


def max_published_cruise(
    pressure_altitude_ft: float, oat_c: float, *, data_dir: Path | None = None
) -> CruisePoint:
    """The most power the POH publishes at this altitude.

    The worst case for level flight: whatever the pilot planned, this is the
    thirstiest setting the book will admit to up there. Used to cost stretches
    of level flight that are not part of the cruise plan -- the tail of a climb
    leg that tops out before the waypoint, where charging the planned economy
    setting would flatter a segment nobody planned.
    """
    usable = available_cruise_rpm(pressure_altitude_ft, oat_c, data_dir=data_dir)
    if not usable:
        raise OutsidePOHEnvelope(
            f"the POH publishes no cruise setting at {pressure_altitude_ft:g} ft"
        )
    return cruise(pressure_altitude_ft, max(usable), oat_c, data_dir=data_dir)


def cruise_isa_band(data_dir: Path | None = None) -> tuple[float, float]:
    """The ISA deviations the cruise chart prints columns for.

    Read off the digitized data rather than written down, so a re-digitized
    chart with more columns widens the band instead of silently disagreeing
    with a constant nobody updated.
    """
    devs = _tables(data_dir).cruise.isa_devs
    return float(devs[0]), float(devs[-1])


def cruise_extrapolation_band(data_dir: Path | None = None) -> tuple[float, float]:
    """The band outside which a cruise reading counts as an extrapolation.

    The published columns, plus `CRUISE_ISA_TOLERANCE_C` of slop either side.
    """
    low, high = cruise_isa_band(data_dir)
    return low - CRUISE_ISA_TOLERANCE_C, high + CRUISE_ISA_TOLERANCE_C


def cruise_altitude_range(data_dir: Path | None = None) -> tuple[float, float]:
    """Lowest and highest pressure altitude the cruise chart covers."""
    alts = _tables(data_dir).cruise.altitudes
    return float(alts[0]), float(alts[-1])


def climb_table_ceiling_ft(data_dir: Path | None = None) -> float:
    """The highest pressure altitude the climb table answers for.

    A "climb to" with no stated altitude means "as high as you get in the
    distance available", so the solver needs something to bisect toward. The
    top published row is the honest target: above it the engine would have to
    extrapolate, which it refuses to do.
    """
    return float(_tables(data_dir).climb_cum_time.axes[0][-1])


def available_cruise_rpm(
    pressure_altitude_ft: float,
    oat_c: float,
    *,
    data_dir: Path | None = None,
    density_substitution: bool = False,
) -> list[float]:
    """Which published RPM settings are usable at this altitude.

    Used by the altitude optimiser, which needs to enumerate real power
    settings rather than guess at one and be refused. Between two altitude
    pages this is what the *lower* page publishes, since that is the only page
    a query there is read from.

    `density_substitution` asks the same question of `cruise_at_density`
    instead: which settings exist where the chart would actually be read for
    this air. Off by default, so that this reports the published chart and
    nothing else unless a caller has decided otherwise.
    """
    table = _tables(data_dir).cruise
    lo, hi = float(min(a[0] for a in table.rpms)), float(max(a[-1] for a in table.rpms))
    read = cruise_at_density if density_substitution else cruise
    usable = []
    for setting in np.arange(lo, hi + 1, 50.0):
        try:
            read(pressure_altitude_ft, float(setting), oat_c, data_dir=data_dir)
        except OutsidePOHEnvelope:
            continue
        usable.append(float(setting))
    return usable
