#!/usr/bin/env python3
"""Plot how far the altimeter setting actually moves over a day, and what it costs.

This exists to answer a question the planner's defaults depend on: if the
altimeter setting barely moves, then planning on 29.92 is close enough and
fetching a live one is a nicety. If it swings, then every takeoff and landing
distance computed on a default setting is wrong, and the surface weather tier
is load bearing rather than decorative.

The second plot is the one that answers it, because it converts the pressure
into the unit the decision is actually made in. A pilot does not care that the
setting is 29.71 rather than 29.92; they care that the aeroplane thinks it is
two hundred feet higher than the field elevation says, and that the runway
required goes up accordingly. So the deviation is drawn twice on shared axes:
in inches, and in the pressure altitude error it produces.

Observations come from the Aviation Weather Center's METAR endpoint, which
takes an `hours` parameter and a comma-separated station list, so the whole
24-hour history for every field is a single request. Decoding is done by
`engine.weather`, not here -- a trend plot that disagreed with the planner
about a unit would be worse than no plot at all.

Usage:
    uv run python tools/plot_altimeter_trend.py
    uv run python tools/plot_altimeter_trend.py --hours 48 --stations KSQL,KSFO
    uv run python tools/plot_altimeter_trend.py --refresh --out build/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import weather as wx
from engine.atmosphere import pressure_altitude

# The Bay Area fields this project is flown from and around. KPAO and KSQL are
# part-time, which is deliberate: their gaps are part of what the plot shows.
DEFAULT_STATIONS = ("KPAO", "KLVK", "KSQL", "KMOD", "KSFO", "KSJC")

AWC_METAR = "https://aviationweather.gov/api/data/metar"
USER_AGENT = "my_e6b VFR planner (+https://github.com/huiyulhy/my_e6b)"
CACHE = ROOT / ".cache" / "wx"
STANDARD_INHG = 29.92126

# Every distance in this program is read at a field, so the pressure altitude
# error is quoted at sea level where the arithmetic is cleanest. The error is
# very nearly independent of elevation anyway -- it is the setting that moves,
# not the lapse rate.
REFERENCE_ELEVATION_FT = 0.0


def fetch_history(stations: tuple[str, ...], hours: int, *, refresh: bool) -> list:
    """Every observation for every station, in one request, cached on disk.

    Cached because the point of this script is to look at the plot, adjust it
    and look again, and re-downloading a megabyte of METARs on each pass is
    rude to a free public service. `--refresh` forces a new download, the same
    flag the tools/build_*.py scripts use.
    """
    query = {"ids": ",".join(stations), "format": "json", "hours": str(hours)}
    url = f"{AWC_METAR}?{urllib.parse.urlencode(query)}"

    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = "-".join(stations)
    path = CACHE / f"history-{stamp}-{hours}h.json"

    # An hour old is fine: METARs are issued hourly, so a fresher copy of a
    # 24-hour window differs by at most one observation per station.
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < 3600:
        print(f"using cached {path.relative_to(ROOT)}")
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    print(f"fetching {hours} h of METARs for {', '.join(stations)} ...")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return payload


def summarise(grouped: dict[str, list]) -> None:
    """Print the numbers behind the plot, so the script is useful without it."""
    print()
    print(f"{'station':>8}  {'obs':>4}  {'min':>7}  {'max':>7}  {'range':>7}  {'PA err':>9}")
    print(f"{'':>8}  {'':>4}  {'inHg':>7}  {'inHg':>7}  {'inHg':>7}  {'ft':>9}")
    print("-" * 52)

    for station in sorted(grouped):
        settings = [
            item.altimeter_inhg
            for item in grouped[station]
            if item.altimeter_inhg is not None
        ]
        if not settings:
            print(f"{station:>8}  {'-':>4}  {'no altimeter settings reported':>40}")
            continue
        low, high = min(settings), max(settings)
        # What the full swing is worth in feet: the difference between the
        # pressure altitude the extremes produce at the same physical place.
        spread_ft = abs(
            pressure_altitude(REFERENCE_ELEVATION_FT, low)
            - pressure_altitude(REFERENCE_ELEVATION_FT, high)
        )
        print(
            f"{station:>8}  {len(settings):>4}  {low:>7.2f}  {high:>7.2f}  "
            f"{high - low:>7.2f}  {spread_ft:>9.0f}"
        )

    missing = [station for station, items in grouped.items() if not items]
    if missing:
        print(f"\nno observations at all: {', '.join(sorted(missing))}")

    _summarise_spread(grouped)


def _summarise_spread(grouped: dict[str, list]) -> None:
    """The widest disagreement between two fields at the same moment.

    This is usually the bigger number, and the more useful one. Drift over a
    day is slow enough that a setting an hour old is roughly right; the spread
    across a planning area is present the whole time. It is the direct
    argument against reading one field's ATIS and using it for the next
    airport down the route.
    """
    hourly: dict[int, dict[str, float]] = {}
    for station, items in grouped.items():
        for item in items:
            if item.altimeter_inhg is None:
                continue
            # Bucket to the hour so observations taken minutes apart compare.
            key = int(item.valid_time.timestamp() // 3600)
            hourly.setdefault(key, {})[station] = item.altimeter_inhg

    worst = None
    for key, settings in hourly.items():
        if len(settings) < 2:
            continue
        low = min(settings.items(), key=lambda pair: pair[1])
        high = max(settings.items(), key=lambda pair: pair[1])
        if worst is None or high[1] - low[1] > worst[0]:
            worst = (high[1] - low[1], key, low, high)

    if worst is None:
        return
    spread, key, low, high = worst
    feet = abs(
        pressure_altitude(REFERENCE_ELEVATION_FT, low[1])
        - pressure_altitude(REFERENCE_ELEVATION_FT, high[1])
    )
    when = datetime.fromtimestamp(key * 3600, tz=UTC)
    print(
        f"\nwidest spread between fields at one time: {spread:.2f} inHg "
        f"({feet:.0f} ft) at {when:%Y-%m-%d %H:00}Z\n"
        f"  {low[0]} {low[1]:.2f} vs {high[0]} {high[1]:.2f}"
    )


def plot(grouped: dict[str, list], out_dir: Path, hours: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    series = {
        station: _with_gaps(
            [item.valid_time for item in items if item.altimeter_inhg is not None],
            [item.altimeter_inhg for item in items if item.altimeter_inhg is not None],
        )
        for station, items in sorted(grouped.items())
    }
    series = {station: pair for station, pair in series.items() if pair[0]}
    if not series:
        print("nothing to plot: no altimeter settings in the response")
        return written

    # --- 1. the settings themselves ---------------------------------------
    figure, axes = plt.subplots(figsize=(11, 6))
    for station, (times, values) in series.items():
        # Plotted with markers as well as a line: a part-time field reports in
        # bursts, and the markers make the gaps legible instead of drawing a
        # straight line through six hours of silence.
        axes.plot(times, values, marker=".", markersize=4, linewidth=1.2, label=station)
    axes.axhline(
        STANDARD_INHG,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"standard {STANDARD_INHG:.2f}",
    )
    axes.set_title(f"Altimeter setting, past {hours} h")
    axes.set_ylabel("altimeter setting (inHg)")
    axes.set_xlabel("time (UTC)")
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes.grid(True, alpha=0.3)
    axes.legend(ncol=4, fontsize="small")
    figure.autofmt_xdate()
    figure.tight_layout()

    path = out_dir / "altimeter_setting.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    # --- 2. what the deviation is worth, in feet --------------------------
    figure, axes = plt.subplots(figsize=(11, 6))
    for station, (times, values) in series.items():
        deviation = [value - STANDARD_INHG for value in values]
        axes.plot(times, deviation, marker=".", markersize=4, linewidth=1.2, label=station)
    axes.axhline(0.0, color="black", linestyle="--", linewidth=1)

    axes.set_title(
        f"Deviation from {STANDARD_INHG:.2f} inHg, and the pressure altitude it causes"
    )
    axes.set_ylabel("deviation (inHg)")
    axes.set_xlabel("time (UTC)")
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes.grid(True, alpha=0.3)
    axes.legend(ncol=4, fontsize="small")

    # The right-hand axis is the whole point of this figure. It is computed
    # with the exact `pressure_altitude` rather than the 1000 ft per inHg rule
    # of thumb, matching how the rest of this program pairs the two: the rule
    # is off by tens of feet at the settings that actually occur.
    right = axes.secondary_yaxis(
        "right",
        functions=(
            _elementwise(
                lambda dev: pressure_altitude(REFERENCE_ELEVATION_FT, STANDARD_INHG + dev)
            ),
            _elementwise(lambda ft: _setting_for_pressure_altitude(ft) - STANDARD_INHG),
        ),
    )
    right.set_ylabel("pressure altitude error at the field (ft)")

    figure.autofmt_xdate()
    figure.tight_layout()
    path = out_dir / "altimeter_deviation.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    return written


# A METAR is issued hourly, so a step of more than this means the station
# stopped reporting rather than that the pressure jumped.
GAP_MIN = 90.0


def _with_gaps(times: list, values: list) -> tuple[list, list]:
    """Break the line where the station stopped reporting.

    KPAO and KSQL close overnight. Joining the last evening observation to the
    first morning one draws a straight, confident line across eight hours of
    no data, which is the one thing a trend plot must not do: the gap is
    information, and the pressure certainly did not move linearly through it.
    `float("nan")` breaks the line while leaving the axis scaling alone.
    """
    if not times:
        return [], []
    out_times: list = [times[0]]
    out_values: list = [values[0]]
    for previous, current, value in zip(times, times[1:], values[1:]):
        if (current - previous).total_seconds() / 60.0 > GAP_MIN:
            out_times.append(previous + (current - previous) / 2)
            out_values.append(float("nan"))
        out_times.append(current)
        out_values.append(value)
    return out_times, out_values


def _elementwise(function):
    """Let a scalar-only engine function serve as a matplotlib axis transform.

    `secondary_yaxis` hands its transforms whole numpy arrays when it places
    ticks, and the engine's `pressure_altitude` is deliberately scalar -- it
    refuses out-of-range input with a comparison that an array cannot answer.
    Mapping over the array here keeps that refusal intact rather than
    loosening the engine to suit a plot.
    """
    import numpy as np

    def wrapper(values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return function(float(array))
        return np.array([function(float(item)) for item in array.ravel()]).reshape(
            array.shape
        )

    return wrapper


def _setting_for_pressure_altitude(pressure_altitude_ft: float) -> float:
    """Invert `pressure_altitude` at the reference elevation.

    Only the secondary axis needs this, and only so matplotlib can place its
    ticks; it is the inverse of a smooth monotonic function over the range of
    settings that occur, so a short bisection is simpler and more obviously
    correct than rearranging the barometric formula.
    """
    low, high = 24.0, 34.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if pressure_altitude(REFERENCE_ELEVATION_FT, middle) > pressure_altitude_ft:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stations",
        default=",".join(DEFAULT_STATIONS),
        help="comma-separated ICAO identifiers",
    )
    parser.add_argument("--hours", type=int, default=24, help="hours of history")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "wx")
    parser.add_argument(
        "--refresh", action="store_true", help="re-download instead of using the cache"
    )
    args = parser.parse_args()

    stations = tuple(
        item.strip().upper() for item in args.stations.split(",") if item.strip()
    )
    if not stations:
        parser.error("no stations given")

    payload = fetch_history(stations, args.hours, refresh=args.refresh)
    grouped = wx.parse_metar_history(payload)
    # Keep every station that was asked for, so one that reported nothing is
    # visibly absent in the summary rather than quietly dropped.
    for station in stations:
        grouped.setdefault(station, [])

    summarise(grouped)
    for path in plot(grouped, args.out, args.hours):
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
