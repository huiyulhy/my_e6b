#!/usr/bin/env python3
"""Plot how linear the POH takeoff table actually is along each axis.

The takeoff chart is a 3-D grid of weight x pressure altitude x temperature.
Whether it is honest to interpolate along an axis -- rather than snap to the
nearest published chart -- depends on how much the underlying curve bends
between published nodes. This script draws that bend three ways:

1. raw curves along each axis, with the straight chord between the endpoints
   overlaid, so curvature is visible by eye
2. the residual: linear interpolation between two published nodes, evaluated
   at the node in between, expressed as a percentage of the published value.
   Positive means the straight line reads LONGER than the chart, which is the
   conservative direction.
3. altitude on a log scale, testing the suspicion that distance grows
   geometrically (a fixed percentage per 1000 ft) rather than linearly.

Usage:
    uv run python tools/plot_linearity.py [--field groundroll] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "poh" / "c172s"
TEMPS = (0, 10, 20, 30, 40)


def load(field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (weights, pressure altitudes, temperatures, values[w, p, t])."""
    with (DATA / "takeoff.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(handle)]
    wts = np.array(sorted({r["wt"] for r in rows}))
    palts = np.array(sorted({r["p_alt"] for r in rows}))
    temps = np.array(TEMPS, dtype=float)
    values = np.zeros((len(wts), len(palts), len(temps)))
    for row in rows:
        i = int(np.searchsorted(wts, row["wt"]))
        j = int(np.searchsorted(palts, row["p_alt"]))
        values[i, j, :] = [row[f"{field}_{int(t)}"] for t in temps]
    return wts, palts, temps, values


def chord_residual(axis: np.ndarray, values: np.ndarray, ax: int) -> np.ndarray:
    """Percent error of a linear chord across each interior node.

    For every interior node k, interpolate linearly between nodes k-1 and k+1
    and compare against the published value at k. Returns an array shaped like
    `values` but with the interpolated axis shortened to the interior nodes.
    """
    moved = np.moveaxis(values, ax, 0)
    out = []
    for k in range(1, len(axis) - 1):
        frac = (axis[k] - axis[k - 1]) / (axis[k + 1] - axis[k - 1])
        predicted = moved[k - 1] + frac * (moved[k + 1] - moved[k - 1])
        out.append((predicted - moved[k]) / moved[k] * 100.0)
    return np.moveaxis(np.array(out), 0, ax)


def plot_axis_curves(ax_plot, x, curves, labels, xlabel, title):
    """Raw values against one axis, with the endpoint chord dashed over each."""
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(curves)))
    for y, label, color in zip(curves, labels, colors):
        ax_plot.plot(x, y, "o-", color=color, label=label, markersize=4, linewidth=1.4)
        ax_plot.plot(
            [x[0], x[-1]], [y[0], y[-1]], "--", color=color, alpha=0.45, linewidth=1.0
        )
    ax_plot.set_xlabel(xlabel)
    ax_plot.set_ylabel("distance (ft)")
    ax_plot.set_title(title, fontsize=10)
    ax_plot.grid(alpha=0.25)
    ax_plot.legend(fontsize=7, frameon=False)


def figure_for(field: str) -> plt.Figure:
    wts, palts, temps, v = load(field)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(
        f"POH takeoff {field}: linearity along each axis "
        f"(dashed = straight chord between endpoints)",
        fontsize=12,
    )

    # --- row 1: raw curves, one axis per column --------------------------
    # Weight, at a few representative altitudes, at 20 C.
    t20 = int(np.searchsorted(temps, 20.0))
    plot_axis_curves(
        axes[0, 0],
        wts,
        [v[:, j, t20] for j in range(0, len(palts), 2)],
        [f"{palts[j]:.0f} ft" for j in range(0, len(palts), 2)],
        "weight (lb)",
        "vs WEIGHT (at 20 C)",
    )
    # Pressure altitude, at each weight, at 20 C.
    plot_axis_curves(
        axes[0, 1],
        palts,
        [v[i, :, t20] for i in range(len(wts))],
        [f"{w:.0f} lb" for w in wts],
        "pressure altitude (ft)",
        "vs PRESSURE ALTITUDE (at 20 C)",
    )
    # Temperature, at each weight, at a mid altitude.
    mid = len(palts) // 2
    plot_axis_curves(
        axes[0, 2],
        temps,
        [v[i, mid, :] for i in range(len(wts))],
        [f"{w:.0f} lb" for w in wts],
        "temperature (C)",
        f"vs TEMPERATURE (at {palts[mid]:.0f} ft)",
    )

    # --- row 2: chord residuals ------------------------------------------
    for col, (axis, name, unit) in enumerate(
        ((wts, "weight", "lb"), (palts, "pressure altitude", "ft"), (temps, "temperature", "C"))
    ):
        ax_plot = axes[1, col]
        if len(axis) < 3:
            ax_plot.text(0.5, 0.5, "needs 3+ nodes", ha="center", va="center")
            ax_plot.set_axis_off()
            continue
        residual = chord_residual(axis, v, col)
        interior = axis[1:-1]
        flat = np.moveaxis(residual, col, 0).reshape(len(interior), -1)
        ax_plot.axhline(0, color="black", linewidth=0.8)
        # Every individual residual as a faint point, plus the envelope.
        for k, node in enumerate(interior):
            ax_plot.plot(
                np.full(flat.shape[1], node), flat[k], ".", color="tab:blue",
                alpha=0.35, markersize=4,
            )
        ax_plot.plot(interior, flat.mean(axis=1), "o-", color="tab:red",
                     linewidth=1.5, label="mean")
        ax_plot.fill_between(interior, flat.min(axis=1), flat.max(axis=1),
                             color="tab:red", alpha=0.12, label="min/max")
        worst = np.abs(flat).max()
        ax_plot.set_xlabel(f"{name} ({unit})")
        ax_plot.set_ylabel("chord error (% of published)")
        ax_plot.set_title(
            f"{name.upper()}: worst |error| = {worst:.2f}%\n"
            f"(positive = linear reads longer = conservative)",
            fontsize=9,
        )
        ax_plot.grid(alpha=0.25)
        ax_plot.legend(fontsize=7, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def figure_log_altitude(field: str) -> plt.Figure:
    """Altitude on a log axis: a straight line here means geometric growth."""
    wts, palts, temps, v = load(field)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(wts)))
    for i, (w, color) in enumerate(zip(wts, colors)):
        for k, t in enumerate(temps):
            left.semilogy(palts, v[i, :, k], "-", color=color, alpha=0.6,
                          linewidth=1.0, label=f"{w:.0f} lb" if k == 0 else None)
    left.set_xlabel("pressure altitude (ft)")
    left.set_ylabel(f"{field} (ft, log scale)")
    left.set_title("straight here = fixed % growth per 1000 ft", fontsize=10)
    left.grid(alpha=0.25, which="both")
    left.legend(fontsize=7, frameon=False)

    ratio = v[:, 1:, :] / v[:, :-1, :]
    for i, (w, color) in enumerate(zip(wts, colors)):
        right.plot(palts[1:], ratio[i].mean(axis=1), "o-", color=color,
                   label=f"{w:.0f} lb", linewidth=1.4, markersize=4)
    right.axhline(ratio.mean(), color="black", linestyle="--", linewidth=0.9,
                  label=f"overall mean {ratio.mean():.3f}")
    right.set_xlabel("pressure altitude (ft)")
    right.set_ylabel("ratio to previous 1000 ft")
    right.set_title(
        f"growth factor per +1000 ft "
        f"({(ratio.mean() - 1) * 100:.1f}% average)", fontsize=10
    )
    right.grid(alpha=0.25)
    right.legend(fontsize=7, frameon=False)
    fig.suptitle(f"POH takeoff {field}: is altitude exponential?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_roundup_cost(field: str) -> plt.Figure:
    """What rounding weight up to the next chart costs versus interpolating."""
    wts, _palts, temps, v = load(field)
    fig, ax_plot = plt.subplots(figsize=(7, 4.5))
    query = np.linspace(wts[0], wts[-1], 400)
    # Interpolated answer at 20 C, averaged over altitude, as the baseline.
    t20 = int(np.searchsorted(temps, 20.0))
    base = v[:, :, t20].mean(axis=1)
    interpolated = np.interp(query, wts, base)
    snapped = np.array([base[int(np.searchsorted(wts, q - 1e-9))] for q in query])
    ax_plot.plot(query, interpolated, label="linear interpolation on weight", linewidth=1.6)
    ax_plot.step(query, snapped, where="post",
                 label="round up to next published chart", linewidth=1.6)
    for w in wts:
        ax_plot.axvline(w, color="grey", linestyle=":", linewidth=0.8)
    ax_plot.set_xlabel("takeoff weight (lb)")
    ax_plot.set_ylabel(f"{field} (ft, averaged over altitude at 20 C)")
    over = (snapped - interpolated) / interpolated * 100
    ax_plot.set_title(
        f"weight handling: rounding up over-reads by up to {over.max():.0f}%",
        fontsize=10,
    )
    ax_plot.grid(alpha=0.25)
    ax_plot.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field", default="both", choices=("groundroll", "clearfifty", "both")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/linearity"))
    args = parser.parse_args()

    fields = ("groundroll", "clearfifty") if args.field == "both" else (args.field,)
    args.out.mkdir(parents=True, exist_ok=True)
    for field in fields:
        for suffix, builder in (
            ("axes", figure_for),
            ("altitude_log", figure_log_altitude),
            ("weight_roundup", figure_roundup_cost),
        ):
            path = args.out / f"takeoff_{field}_{suffix}.png"
            builder(field).savefig(path, dpi=130)
            plt.close("all")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
