"""Pre-render every raster chart into Web Mercator tiles.

    make charts
    make charts MAXZOOM=11

This is what puts charts on a deployed server. The dev shim renders a tile
the first time the map asks for it, so nothing here is needed to *see* a
chart while developing -- but a deployed instance has no GeoTIFFs on it (at
30-80 MB each they are not committed) and not enough memory to decode one
if it did. So the pyramid is built here, committed, and served as static
files. See docs/ARCHITECTURE.md section 4e.

Alongside the tiles goes `manifest.json`: each chart's footprint, zoom range
and edition dates, which is everything the layer menu and the currency list
need and would otherwise have read out of the GeoTIFF headers.

Tiles already on disk are kept, so re-running after adding one chart renders
only that chart. `--force` discards them first, which is what to use after
changing how a tile is drawn.

    tools/build_charts.py                    # everything, at native zoom
    tools/build_charts.py tac                # one series
    tools/build_charts.py --max-zoom 11      # cap the pyramid
    tools/build_charts.py --force            # re-render from scratch
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import chart_render as render  # noqa: E402
from engine import charts  # noqa: E402


def _flag_value(argv: list[str], name: str) -> str | None:
    """`--name value` or `--name=value`, or None."""
    for index, argument in enumerate(argv):
        if argument == name and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None


def _tile_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*.png"))


def main(argv: list[str]) -> int:
    force = "--force" in argv
    cap = _flag_value(argv, "--max-zoom")
    max_zoom = int(cap) if cap else None
    skip = {"--force", "--max-zoom", cap}
    wanted = {arg for arg in argv if not arg.startswith("--") and arg not in skip}

    found = charts.discover()
    if wanted:
        found = [chart for chart in found if chart.kind in wanted or chart.slug in wanted]
    if not found:
        matching = f" matching {sorted(wanted)}" if wanted else ""
        print(f"no charts under {charts.CHARTS_DIR}{matching}")
        print("See data/charts/README.txt for where to get them.")
        return 1

    # The manifest must describe what is actually on disk, so a capped chart
    # is capped here, before anything is written. A source that advertises a
    # zoom it does not have would leave holes in the map.
    if max_zoom is not None:
        found = [
            dataclasses.replace(chart, max_zoom=min(chart.max_zoom, max_zoom)) for chart in found
        ]

    total = 0
    started = time.time()
    for chart in found:
        tiles_dir = chart.tiles_dir()
        if force and tiles_dir.exists():
            shutil.rmtree(tiles_dir)
        expected = sum(
            (x1 - x0 + 1) * (y1 - y0 + 1)
            for x0, y0, x1, y1 in (
                render.tile_range(chart.bounds, zoom)
                for zoom in range(chart.min_zoom, chart.max_zoom + 1)
            )
        )
        capped = " (capped)" if max_zoom is not None and max_zoom < chart.max_zoom else ""
        print(
            f"{chart.key}: {chart.name}, zoom {chart.min_zoom}-{chart.max_zoom}{capped}, "
            f"{expected} tiles -> {tiles_dir.relative_to(ROOT)}",
            flush=True,
        )
        count = chart.store().render_all(
            chart.bounds,
            chart.min_zoom,
            chart.max_zoom,
            progress=lambda zoom, n: print(f"  z{zoom}: {n} tiles", flush=True),
        )
        print(f"  {count} tiles, {_tile_bytes(tiles_dir) / 1e6:.1f} MB on disk", flush=True)
        total += count

    # Written last: a manifest is a promise that the tiles under it exist, so
    # every chart's zoom range is read back off the directories rather than
    # taken from what was asked for. Every chart with tiles goes in, not just
    # the ones rebuilt this run, or naming one series on the command line
    # would drop the others out of the layer menu.
    on_disk = []
    for chart in charts.discover():
        zooms = charts.rendered_zooms(chart)
        if zooms is None:
            print(f"{chart.key}: no tiles on disk, left out of the manifest")
            continue
        on_disk.append(dataclasses.replace(chart, min_zoom=zooms[0], max_zoom=zooms[1]))
    manifest = charts.write_manifest(on_disk)
    print(
        f"manifest: {len(on_disk)} charts -> {manifest.relative_to(ROOT)} "
        + ", ".join(f"{c.slug} z{c.min_zoom}-{c.max_zoom}" for c in on_disk)
    )

    size = _tile_bytes(charts.TILES_DIR) / 1e6
    print(f"{total} tiles this run, {size:.1f} MB total, {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
