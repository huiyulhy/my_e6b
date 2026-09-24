"""Render a Lambert-projected chart into Web Mercator map tiles.

MapLibre draws raster imagery one way: 256-pixel tiles in the Web Mercator
"XYZ" scheme. An FAA chart is a single Lambert Conformal Conic image, so
the two never line up and something has to resample one into the other.
This module is that something.

The approach is the plain one. For every pixel of an output tile, work out
its longitude and latitude, project that into the chart's own coordinates
(`tiff_reader.Georeference.to_pixel`), and take the nearest chart pixel.
All of it is vectorised in numpy, so a tile is a few milliseconds. Pixels
that land outside the chart image are transparent, which is how the edge of
a chart shows the basemap beneath.

Nearest-neighbour is right at the chart's own scale -- the linework stays
crisp -- but zoomed far out it would drop thin lines and alias the rest. So
a box-filtered pyramid is built lazily (`ChartRaster.level`) and each tile
is sampled from the level whose pixels are closest in size to its own.

Tiles are cached on disk as PNG under `data/charts/tiles/` by `TileStore`,
which is what both the dev server and `tools/build_charts.py` go through:
the server fills the cache on demand while developing, the tool fills all
of it for the offline bundle, and neither renders a tile twice.
"""

from __future__ import annotations

import io
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine import tiff_reader as tr

TILE_SIZE = 256
# Web Mercator: the sphere it is defined on, and the latitude it stops at.
MERCATOR_RADIUS_M = 6378137.0
MAX_MERCATOR_LAT = 85.05112878
# Ground metres per pixel at zoom 0 on the equator, for a 256 px tile.
ZOOM_0_METRES_PER_PIXEL = 2 * math.pi * MERCATOR_RADIUS_M / TILE_SIZE

# Below this the whole sectional is a few pixels and the tiles are noise.
MIN_ZOOM = 5
# Pyramid levels stop once the chart is smaller than a couple of tiles.
MIN_LEVEL_SIZE_PX = 2 * TILE_SIZE
# Level 1 is built in bands so that the full-resolution RGB image, three
# times the size of the index array, never exists all at once.
BAND_ROWS = 1024

PNG_MEDIA_TYPE = "image/png"
# Colours a tile's palette may hold. One short of 256, so there is always a
# free index left over to mean "transparent".
PALETTE_COLOURS = 255


# --- tile arithmetic ---------------------------------------------------------


def metres_per_pixel(zoom: int, lat_deg: float) -> float:
    """Ground resolution of a tile pixel at this zoom and latitude."""
    return ZOOM_0_METRES_PER_PIXEL * math.cos(math.radians(lat_deg)) / 2**zoom


def native_zoom(chart_metres_per_pixel: float, lat_deg: float) -> int:
    """The first zoom at which tile pixels are no coarser than chart pixels.

    Rounding *up* means the chart's finest linework survives; one level
    higher than that would only be the same pixels magnified.
    """
    ratio = ZOOM_0_METRES_PER_PIXEL * math.cos(math.radians(lat_deg)) / chart_metres_per_pixel
    return max(MIN_ZOOM, math.ceil(math.log2(ratio)))


def tile_at(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """The (x, y) tile containing a point."""
    n = 2**zoom
    lat = max(-MAX_MERCATOR_LAT, min(MAX_MERCATOR_LAT, lat))
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_range(bounds: tuple[float, float, float, float], zoom: int) -> tuple[int, int, int, int]:
    """Inclusive (x0, y0, x1, y1) of the tiles covering a lon/lat box."""
    west, south, east, north = bounds
    x0, y0 = tile_at(west, north, zoom)
    x1, y1 = tile_at(east, south, zoom)
    return x0, y0, x1, y1


def tile_pixel_lonlat(zoom: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude at the centre of every pixel in a tile.

    Returned as two (TILE_SIZE, TILE_SIZE) arrays.
    """
    n = 2**zoom
    span = 1.0 / n
    offsets = (np.arange(TILE_SIZE) + 0.5) / TILE_SIZE
    fx = (x + offsets) * span  # 0..1 across the world, west to east
    fy = (y + offsets) * span  # 0..1 down the world, north to south
    lon = fx * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * fy))))
    return np.broadcast_to(lon, (TILE_SIZE, TILE_SIZE)), np.broadcast_to(
        lat[:, None], (TILE_SIZE, TILE_SIZE)
    )


# --- the decoded chart ------------------------------------------------------


@dataclass
class ChartRaster:
    """One chart, decoded, with the pyramid it is sampled from.

    `indices` is the palette image as stored; `palette` maps it to colour.
    Levels 1 and up are RGB, halved in size each step, and built the first
    time a tile needs them.
    """

    header: tr.GeoTiff
    indices: np.ndarray
    palette: np.ndarray  # (256, 3) uint8
    _levels: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def load(cls, path: Path) -> ChartRaster:
        header = tr.read_header(path)
        if header.georeference is None:
            raise tr.NotGeoreferenced(f"{Path(path).name}: no georeferencing")
        if not header.is_palette:
            raise tr.TiffError(f"{Path(path).name}: only palette charts are supported")
        palette = np.array(header.palette, dtype=np.uint8)
        if palette.shape[0] < 256:
            palette = np.vstack([palette, np.zeros((256 - palette.shape[0], 3), np.uint8)])
        return cls(header, tr.read_pixels(path), palette)

    @property
    def georeference(self) -> tr.Georeference:
        assert self.header.georeference is not None
        return self.header.georeference

    @property
    def max_level(self) -> int:
        smallest = min(self.indices.shape)
        level = 0
        while smallest / 2 ** (level + 1) >= MIN_LEVEL_SIZE_PX:
            level += 1
        return level

    def level(self, k: int) -> np.ndarray:
        """RGB pixels at 1/2**k scale, k >= 1. Built and kept on first use."""
        if k < 1:
            raise ValueError("level 0 is the index array; use `indices` and `palette`")
        with self._lock:
            if k in self._levels:
                return self._levels[k]
            image = self._halve_indices() if k == 1 else self._halve_rgb(self.level(k - 1))
            self._levels[k] = image
            return image

    def _halve_indices(self) -> np.ndarray:
        from PIL import Image

        height, width = self.indices.shape
        out = np.empty((-(-height // 2), -(-width // 2), 3), dtype=np.uint8)
        for top in range(0, height, BAND_ROWS):
            band = self.palette[self.indices[top : top + BAND_ROWS]]
            reduced = np.asarray(Image.fromarray(band, "RGB").reduce(2))
            out[top // 2 : top // 2 + reduced.shape[0]] = reduced
        return out

    @staticmethod
    def _halve_rgb(rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.fromarray(rgb, "RGB").reduce(2))

    def level_for(self, zoom: int, lat_deg: float) -> int:
        """Which pyramid level has pixels nearest the size of this tile's."""
        ratio = metres_per_pixel(zoom, lat_deg) / self.georeference.pixel_width_m
        if ratio <= 1.0:
            return 0
        return min(self.max_level, int(math.floor(math.log2(ratio))))

    def render_tile(self, zoom: int, x: int, y: int) -> np.ndarray | None:
        """An RGBA (TILE_SIZE, TILE_SIZE, 4) array, or None if the tile is empty."""
        lon, lat = tile_pixel_lonlat(zoom, x, y)
        col, row = self.georeference.to_pixel(lon, lat)
        height, width = self.indices.shape
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        if not inside.any():
            return None

        k = self.level_for(zoom, float(lat[TILE_SIZE // 2, 0]))
        scale = 2**k
        # Clamp before indexing so the out-of-image pixels index *something*;
        # their alpha is zeroed below, so what they read does not matter.
        c = np.clip(np.floor(col / scale).astype(np.int64), 0, -(-width // scale) - 1)
        r = np.clip(np.floor(row / scale).astype(np.int64), 0, -(-height // scale) - 1)
        if k == 0:
            rgb = self.palette[self.indices[r, c]]
        else:
            rgb = self.level(k)[r, c]

        tile = np.empty((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
        tile[..., :3] = rgb
        tile[..., 3] = np.where(inside, 255, 0)
        return tile


def png_bytes(rgba: np.ndarray) -> bytes:
    """Encode a tile as a *palette* PNG with one transparent index.

    The charts are 8-bit palette images, so a tile cut from one carries only
    a handful of distinct colours and an indexed PNG is both exact and about
    a third the size of RGBA -- which matters when the whole pyramid is
    committed and served as static files.

    Two cases. A tile sampled at the chart's own scale holds only palette
    colours, so an exact palette is built from the colours actually present
    and nothing is lost. A tile sampled from a box-filtered pyramid level
    holds blended colours, thousands of them, and is quantised to 255 --
    imperceptible on an image that is already a reduction, and the only
    lossy step anywhere in this module.
    """
    from PIL import Image

    opaque = (rgba[..., 3] == 255).reshape(-1)
    flat = rgba[..., :3].reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    colours = np.unique(packed[opaque])

    if colours.size <= PALETTE_COLOURS:
        indices = np.searchsorted(colours, packed).astype(np.uint8)
        palette = np.stack(
            [colours >> 16 & 0xFF, colours >> 8 & 0xFF, colours & 0xFF], axis=1
        ).astype(np.uint8)
    else:
        quantised = (
            Image.fromarray(rgba, "RGBA")
            .convert("RGB")
            .quantize(colors=PALETTE_COLOURS, method=Image.Quantize.FASTOCTREE)
        )
        # asarray over a PIL image is read-only; the transparent index is
        # written into this below.
        indices = np.array(quantised, dtype=np.uint8).reshape(-1)
        palette = np.array(quantised.getpalette(), dtype=np.uint8).reshape(-1, 3)
        palette = palette[:PALETTE_COLOURS]

    # One index past the colours in use is the transparent one. Every tile
    # has room for it: neither branch above ever fills all 256 entries.
    transparent = len(palette)
    indices[~opaque] = transparent
    # The palette is written no longer than it needs to be. Most tiles of a
    # chart hold a handful of colours, and a full 256-entry table would be
    # 768 bytes of padding on each of them -- several megabytes across a
    # pyramid, on tiles that are otherwise under a kilobyte.
    table = np.vstack([palette, np.zeros((1, 3), np.uint8)])

    image = Image.fromarray(indices.reshape(TILE_SIZE, TILE_SIZE), "P")
    image.putpalette(table.tobytes())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=transparent)
    return buffer.getvalue()


def _empty_png() -> bytes:
    return png_bytes(np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8))


EMPTY_TILE_PNG = _empty_png()


# --- the on-disk cache ------------------------------------------------------


class TileStore:
    """PNG tiles for one chart under `<root>/{z}/{x}/{y}.png`, rendered on miss.

    The decoded chart is loaded the first time a tile has to be rendered,
    and not at all if every tile asked for is already on disk -- so a
    pre-rendered bundle never pays the decode.
    """

    def __init__(self, chart_path: Path, root: Path) -> None:
        self.chart_path = Path(chart_path)
        self.root = Path(root)
        self._raster: ChartRaster | None = None
        self._lock = threading.Lock()

    @property
    def raster(self) -> ChartRaster:
        with self._lock:
            if self._raster is None:
                self._raster = ChartRaster.load(self.chart_path)
            return self._raster

    def tile_path(self, zoom: int, x: int, y: int) -> Path:
        return self.root / str(zoom) / str(x) / f"{y}.png"

    def get(self, zoom: int, x: int, y: int) -> bytes:
        """The tile's PNG bytes, from disk if it is there, else rendered and kept."""
        path = self.tile_path(zoom, x, y)
        try:
            return path.read_bytes()
        except OSError:
            pass
        rgba = self.raster.render_tile(zoom, x, y)
        data = EMPTY_TILE_PNG if rgba is None else png_bytes(rgba)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a concurrent reader never sees a half tile.
        temporary = path.with_suffix(".png.part")
        temporary.write_bytes(data)
        temporary.replace(path)
        return data

    def render_all(
        self,
        bounds: tuple[float, float, float, float],
        min_zoom: int,
        max_zoom: int,
        progress=None,
    ) -> int:
        """Fill the cache for every tile touching `bounds`. Returns the count."""
        count = 0
        for zoom in range(min_zoom, max_zoom + 1):
            x0, y0, x1, y1 = tile_range(bounds, zoom)
            written = 0
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    self.get(zoom, x, y)
                    written += 1
            count += written
            if progress:
                progress(zoom, written)
        return count
