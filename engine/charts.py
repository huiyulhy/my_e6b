"""Which charts are on disk, where each one sits, and how long it is good for.

Charts live under `data/charts/<kind>/<any folder>/<name>.tif`, exactly as
the FAA ships them: one folder per download, a `.tif` beside a `.tfw` and an
`.htm`. The first level -- `sectional`, `tac` -- is the series, and is what
the map's layer menu groups by. Everything below it is walked, so dropping a
new download into the right folder is all it takes to add a chart.

Two facts come off each chart:

* **Placement** from the GeoTIFF tags (`tiff_reader`): its footprint in
  longitude/latitude, and the zoom range it is worth drawing at.
* **Currency** from the FAA's FGDC metadata, the `.htm` beside the image.
  It states the edition's `Beginning_Date` and `Ending_Date`; VFR charts
  run on a 56-day cycle, and the ending date is the last day the chart
  is legal to fly with. That is what the currency list shows.

Charts reach the app two ways, and which one is in play depends on whether
the GeoTIFFs are on the machine:

* **From the files themselves**, on the desktop, where `discover` reads the
  headers and tiles are rendered on demand.
* **From `tiles/manifest.json`**, on a deployed server, where there are no
  GeoTIFFs at all -- only the pyramid `make charts` wrote, served as static
  files. The manifest carries what the headers would have said, so the layer
  menu and the currency list read the same either way.

`available` picks between them. Nothing here decodes pixels: discovery reads
a few hundred bytes of header per chart, so listing the folder is cheap
enough to do on every request.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from engine import chart_render as render
from engine import tiff_reader as tr

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHARTS_DIR = DATA_DIR / "charts"
# Rendered tiles go beside the source charts. Unlike the charts themselves
# these *are* committed: they are the only form in which a deployed server,
# which has no GeoTIFFs, can serve a chart. `make charts` writes them.
TILES_DIR = CHARTS_DIR / "tiles"
MANIFEST_NAME = "manifest.json"
# Bumped if the manifest's shape changes; an older file is then ignored
# rather than half-read.
MANIFEST_VERSION = 1

# Folder name -> what the layer menu calls it, in the order the menu lists
# them, which is also draw order: a TAC is more detailed than the sectional
# under it, so it goes on top. Folders not listed here are still found, shown
# after these under their own name.
KINDS: dict[str, str] = {
    "sectional": "Sectional",
    "tac": "Terminal Area",
}


class NoChartFile(FileNotFoundError):
    """The chart's GeoTIFF is not on this machine."""


_SLUG = re.compile(r"[^a-z0-9]+")
_TAG = re.compile(r"<[^>]+>")
_DATE_FIELD = {
    "begin": re.compile(r"Beginning_Date:\s*(\d{8})"),
    "end": re.compile(r"Ending_Date:\s*(\d{8})"),
    "published": re.compile(r"Publication_Date:\s*(\d{8})"),
}
_ROWS = re.compile(r"Row_Count:\s*(\d+)")
_COLUMNS = re.compile(r"Column_Count:\s*(\d+)")


@dataclass(frozen=True)
class Edition:
    """What the FAA metadata file says about the chart's dates and size."""

    begins: date | None
    ends: date | None
    published: date | None
    rows: int | None
    columns: int | None


@dataclass(frozen=True)
class Chart:
    kind: str  # the folder under data/charts: "sectional", "tac"
    slug: str  # url-safe, unique within its kind: "san_francisco_sec"
    name: str  # the file's own name: "San Francisco SEC"
    # None for a chart read from the manifest: the tiles are on disk but the
    # GeoTIFF they came from is not, so nothing can be rendered from it.
    path: Path | None
    width: int
    height: int
    pixel_m: float
    bounds: tuple[float, float, float, float]  # west, south, east, north
    min_zoom: int
    max_zoom: int
    effective: date | None
    expires: date | None
    note: str | None = None
    # True when the tiles already exist and no renderer is involved.
    prerendered: bool = False

    @property
    def kind_label(self) -> str:
        return KINDS.get(self.kind, self.kind.replace("_", " ").title())

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.slug}"

    def tiles_dir(self, tiles_root: Path | None = None) -> Path:
        return (tiles_root or TILES_DIR) / self.kind / self.slug

    def store(self, tiles_root: Path | None = None) -> render.TileStore:
        if self.path is None:
            raise NoChartFile(
                f"{self.key} was read from the tile manifest; its GeoTIFF is not on "
                f"this machine, so its tiles can only be served, not rendered."
            )
        return render.TileStore(self.path, self.tiles_dir(tiles_root))


def slugify(name: str) -> str:
    return _SLUG.sub("_", name.lower()).strip("_")


def _fgdc_date(text: str, pattern: re.Pattern) -> date | None:
    match = pattern.search(text)
    if not match:
        return None
    raw = match.group(1)
    try:
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return None


def read_edition(htm_path: Path) -> Edition | None:
    """Dates and image size from the FGDC `.htm` the FAA ships with each chart.

    The file is HTML wrapped around a plain outline, so stripping the tags
    leaves `Field_Name: value` lines. The first `Beginning_Date` in the file
    is the edition's own (the source-data dates repeat it further down).
    """
    try:
        raw = Path(htm_path).read_text(errors="replace")
    except OSError:
        return None
    text = html.unescape(_TAG.sub(" ", raw))
    rows = _ROWS.search(text)
    columns = _COLUMNS.search(text)
    return Edition(
        begins=_fgdc_date(text, _DATE_FIELD["begin"]),
        ends=_fgdc_date(text, _DATE_FIELD["end"]),
        published=_fgdc_date(text, _DATE_FIELD["published"]),
        rows=int(rows.group(1)) if rows else None,
        columns=int(columns.group(1)) if columns else None,
    )


def _kind_order(kind: str) -> tuple[int, str]:
    known = list(KINDS)
    return (known.index(kind) if kind in known else len(known), kind)


def _describe(kind: str, path: Path) -> Chart:
    header = tr.read_header(path)
    name = path.stem
    if header.georeference is None:
        raise tr.NotGeoreferenced(f"{path.name}: no georeferencing")
    geo = header.georeference
    bounds = geo.lonlat_bounds(header.width, header.height)
    centre_lat = (bounds[1] + bounds[3]) / 2

    edition = read_edition(path.with_suffix(".htm"))
    note = None
    if edition is None:
        note = f"No FAA metadata file ({path.stem}.htm) beside the chart; edition dates unknown."
        effective = expires = None
    else:
        effective, expires = edition.begins, edition.ends
        if effective is None and expires is None:
            note = f"{path.stem}.htm has no edition dates."
        elif (edition.rows, edition.columns) != (header.height, header.width) and (
            edition.rows is not None or edition.columns is not None
        ):
            note = (
                f"Metadata describes a {edition.columns}x{edition.rows} image; "
                f"the file is {header.width}x{header.height}. The .htm and .tif "
                f"may be from different editions."
            )

    return Chart(
        kind=kind,
        slug=slugify(name),
        name=name,
        path=path,
        width=header.width,
        height=header.height,
        pixel_m=geo.pixel_width_m,
        bounds=bounds,
        min_zoom=render.MIN_ZOOM,
        max_zoom=render.native_zoom(geo.pixel_width_m, centre_lat),
        effective=effective,
        expires=expires,
        note=note,
    )


def discover(charts_dir: Path | None = None) -> list[Chart]:
    """Every readable chart under `data/charts`, series by series.

    A file that is not a GeoTIFF this reader understands is skipped rather
    than fatal: one bad download should not take the layer menu down.
    """
    root = Path(charts_dir) if charts_dir else CHARTS_DIR
    if not root.is_dir():
        return []
    charts: list[Chart] = []
    for kind_dir in sorted(root.iterdir(), key=lambda p: _kind_order(p.name)):
        if not kind_dir.is_dir() or kind_dir == (root / TILES_DIR.name):
            continue
        seen: set[str] = set()
        for path in sorted(kind_dir.rglob("*.tif")) + sorted(kind_dir.rglob("*.tiff")):
            try:
                chart = _describe(kind_dir.name, path)
            except (tr.TiffError, OSError):
                continue
            if chart.slug in seen:
                continue
            seen.add(chart.slug)
            charts.append(chart)
    return charts


def find(kind: str, slug: str, charts_dir: Path | None = None) -> Chart | None:
    for chart in discover(charts_dir):
        if chart.kind == kind and chart.slug == slug:
            return chart
    return None


# --- the manifest -----------------------------------------------------------
#
# What a deployed server knows about a chart. Everything in `Chart` that does
# not need the GeoTIFF, written beside the tiles so the two travel together.


def manifest_path(tiles_root: Path | None = None) -> Path:
    return (tiles_root or TILES_DIR) / MANIFEST_NAME


def _manifest_entry(chart: Chart) -> dict:
    return {
        "kind": chart.kind,
        "slug": chart.slug,
        "name": chart.name,
        "width": chart.width,
        "height": chart.height,
        "pixel_m": chart.pixel_m,
        "bounds": list(chart.bounds),
        "min_zoom": chart.min_zoom,
        "max_zoom": chart.max_zoom,
        "effective": chart.effective.isoformat() if chart.effective else None,
        "expires": chart.expires.isoformat() if chart.expires else None,
        "note": chart.note,
    }


def _chart_from_entry(entry: dict) -> Chart:
    def parse(key: str) -> date | None:
        raw = entry.get(key)
        return date.fromisoformat(raw) if raw else None

    return Chart(
        kind=entry["kind"],
        slug=entry["slug"],
        name=entry["name"],
        path=None,
        width=int(entry["width"]),
        height=int(entry["height"]),
        pixel_m=float(entry["pixel_m"]),
        bounds=tuple(entry["bounds"]),
        min_zoom=int(entry["min_zoom"]),
        max_zoom=int(entry["max_zoom"]),
        effective=parse("effective"),
        expires=parse("expires"),
        note=entry.get("note"),
        prerendered=True,
    )


def rendered_zooms(chart: Chart, tiles_root: Path | None = None) -> tuple[int, int] | None:
    """The zoom levels this chart actually has tiles for, or None if it has none.

    Read off the directory names rather than assumed, because the pyramid on
    disk is not always the chart's native range: `make charts --max-zoom`
    caps it to keep the committed tree small, and the two series are often
    capped differently. A manifest that advertised a level with no tiles
    behind it would show as holes in the map.
    """
    directory = chart.tiles_dir(tiles_root)
    zooms = (
        sorted(
            int(child.name)
            for child in directory.iterdir()
            if child.is_dir() and child.name.isdigit()
        )
        if directory.is_dir()
        else []
    )
    return (zooms[0], zooms[-1]) if zooms else None


def write_manifest(charts: list[Chart], tiles_root: Path | None = None) -> Path:
    """Record what was rendered, so a server without the GeoTIFFs can list it."""
    path = manifest_path(tiles_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "version": MANIFEST_VERSION,
        "generated": datetime.now(tz=UTC).date().isoformat(),
        "charts": [_manifest_entry(chart) for chart in charts],
    }
    path.write_text(json.dumps(body, indent=2) + "\n")
    return path


def read_manifest(tiles_root: Path | None = None) -> list[Chart]:
    """The charts the manifest describes, or none if it is absent or stale."""
    try:
        body = json.loads(manifest_path(tiles_root).read_text())
    except (OSError, ValueError):
        return []
    if body.get("version") != MANIFEST_VERSION:
        return []
    charts: list[Chart] = []
    for entry in body.get("charts", []):
        try:
            charts.append(_chart_from_entry(entry))
        except (KeyError, TypeError, ValueError):
            continue
    return charts


def available(charts_dir: Path | None = None, tiles_root: Path | None = None) -> list[Chart]:
    """Every chart this machine can show, from the GeoTIFFs or the manifest.

    The files win where they exist: on the desktop a chart can be rendered at
    any zoom on demand, so a capped pyramid should not limit what the map
    draws. A deployed server has no files and falls back to the manifest.
    """
    return discover(charts_dir) or read_manifest(tiles_root)
