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

import contextlib
import dataclasses
import errno
import fcntl
import os
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


@contextlib.contextmanager
def _only_one_build():
    """Refuse to start if another build is already running.

    Two builds over one tile tree corrupt each other: the second deletes a
    chart's directory to re-render it while the first is still writing into
    it, and the delete fails half way with `Directory not empty`, leaving a
    tree that is neither edition. Cheap to prevent, confusing to debug.
    """
    charts.TILES_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = charts.TILES_DIR / ".build.lock"
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise SystemExit(
                f"another chart build is already running (lock: {lock_path}).\n"
                f"Wait for it to finish, or stop it, then run this again."
            ) from exc
        yield
    finally:
        os.close(handle)


def _clear(directory: Path) -> None:
    """Delete a chart's tiles, tolerating a file appearing mid-walk."""
    for attempt in range(3):
        try:
            shutil.rmtree(directory)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.2)


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
        # Only an error if a specific chart was asked for. With none
        # downloaded at all this is a dataset rebuild on a checkout that has
        # no charts yet, which is a normal state, not a failure.
        return 1 if wanted else 0

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
        # A new edition lands at the same path with the same tile filenames,
        # and tiles already on disk are kept -- so without this the previous
        # edition's pixels would survive under the new edition's dates.
        stale = charts.stale_reason(chart)
        if stale:
            print(f"{chart.key}: re-rendering, {stale}", flush=True)
        if (force or stale) and tiles_dir.exists():
            _clear(tiles_dir)
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
    with _only_one_build():
        raise SystemExit(main(sys.argv[1:]))
