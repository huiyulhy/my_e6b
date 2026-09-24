"""Read an FAA raster chart GeoTIFF: its tags, its georeferencing, its pixels.

The FAA publishes every VFR chart (Sectional, Terminal Area, Flyway) as a
single 8-bit palette GeoTIFF in a Lambert Conformal Conic projection. A
GeoTIFF is an ordinary TIFF with a few extra tags that say where the image
sits on the earth: the size of a pixel in map units, one tie point pinning
a pixel to a map coordinate, and a small key directory naming the projection
and its parameters. This module parses those tags directly -- they are a
handful of `struct` calls and needing GDAL for them would be a heavy
dependency for a few hundred bytes of header.

Decoding the LZW-compressed pixels is left to Pillow, which is fast enough
that even the 200-megapixel sectional opens in about a second.

Two coordinate systems meet here:

* **map** coordinates (x, y) in metres on the chart's own projection, and
* **pixel** coordinates (col, row), with (0, 0) the top-left corner of the
  top-left pixel (GeoTIFF `RasterPixelIsArea`).

`Georeference` converts between those two and geographic longitude/latitude.
NAD83 and WGS84 differ by under two metres anywhere in the continental US,
which is far below a chart pixel (21-42 m), so the datums are treated as one.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- TIFF tags ------------------------------------------------------------

TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_SAMPLES_PER_PIXEL = 277
TAG_DATETIME = 306
TAG_COLOR_MAP = 320
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_MODEL_TRANSFORMATION = 34264
TAG_GEO_KEY_DIRECTORY = 34735
TAG_GEO_DOUBLE_PARAMS = 34736
TAG_GEO_ASCII_PARAMS = 34737

PHOTOMETRIC_PALETTE = 3

# Bytes per element for each TIFF field type.
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8}
_TYPE_FORMAT = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d", 16: "Q"}

# --- GeoTIFF keys -----------------------------------------------------------

KEY_MODEL_TYPE = 1024
KEY_RASTER_TYPE = 1025
KEY_CITATION = 1026
KEY_GEOGRAPHIC_TYPE = 2048
KEY_SEMI_MAJOR_AXIS = 2057
KEY_INV_FLATTENING = 2059
KEY_PROJECTED_CS_TYPE = 3072
KEY_PROJ_COORD_TRANS = 3075
KEY_PROJ_LINEAR_UNITS = 3076
KEY_STD_PARALLEL_1 = 3078
KEY_STD_PARALLEL_2 = 3079
KEY_NAT_ORIGIN_LONG = 3080
KEY_NAT_ORIGIN_LAT = 3081
KEY_FALSE_EASTING = 3082
KEY_FALSE_NORTHING = 3083
KEY_FALSE_ORIGIN_LONG = 3084
KEY_FALSE_ORIGIN_LAT = 3085
KEY_FALSE_ORIGIN_EASTING = 3086
KEY_FALSE_ORIGIN_NORTHING = 3087

MODEL_TYPE_PROJECTED = 1
RASTER_PIXEL_IS_AREA = 1
RASTER_PIXEL_IS_POINT = 2
COORD_TRANS_LCC_2SP = 8
COORD_TRANS_LCC_1SP = 9
LINEAR_UNIT_METRE = 9001

# GRS80 / WGS84 -- the two are identical to the precision that matters here.
DEFAULT_SEMI_MAJOR_M = 6378137.0
DEFAULT_INV_FLATTENING = 298.257222101


class TiffError(ValueError):
    """The file is not a TIFF this reader understands."""


class NotGeoreferenced(TiffError):
    """The TIFF carries no usable georeferencing."""


# --- the projection ---------------------------------------------------------


@dataclass(frozen=True)
class LambertConformalConic:
    """Lambert Conformal Conic on an ellipsoid, two standard parallels.

    Snyder, *Map Projections: A Working Manual*, equations 15-1 to 15-11.
    A one-parallel chart is the same thing with both parallels equal.
    Angles are degrees; `forward` returns metres; both accept scalars or
    numpy arrays.
    """

    standard_parallel_1: float
    standard_parallel_2: float
    central_meridian: float
    latitude_of_origin: float
    false_easting: float = 0.0
    false_northing: float = 0.0
    semi_major_m: float = DEFAULT_SEMI_MAJOR_M
    inverse_flattening: float = DEFAULT_INV_FLATTENING

    @property
    def eccentricity(self) -> float:
        flattening = 1.0 / self.inverse_flattening
        return math.sqrt(2 * flattening - flattening * flattening)

    def _m(self, phi):
        e = self.eccentricity
        return np.cos(phi) / np.sqrt(1 - (e * np.sin(phi)) ** 2)

    def _t(self, phi):
        e = self.eccentricity
        es = e * np.sin(phi)
        return np.tan(math.pi / 4 - phi / 2) / ((1 - es) / (1 + es)) ** (e / 2)

    def _constants(self) -> tuple[float, float, float]:
        """(n, F, rho_0) -- the cone constant, and the scale terms."""
        phi1 = math.radians(self.standard_parallel_1)
        phi2 = math.radians(self.standard_parallel_2)
        phi0 = math.radians(self.latitude_of_origin)
        m1, t1 = float(self._m(phi1)), float(self._t(phi1))
        if abs(phi1 - phi2) < 1e-10:
            n = math.sin(phi1)
        else:
            m2, t2 = float(self._m(phi2)), float(self._t(phi2))
            n = (math.log(m1) - math.log(m2)) / (math.log(t1) - math.log(t2))
        big_f = m1 / (n * t1**n)
        rho0 = self.semi_major_m * big_f * float(self._t(phi0)) ** n
        return n, big_f, rho0

    def forward(self, lon, lat):
        """Longitude/latitude in degrees to map (x, y) in metres."""
        n, big_f, rho0 = self._constants()
        phi = np.radians(np.asarray(lat, dtype=float))
        lam = np.radians(np.asarray(lon, dtype=float))
        rho = self.semi_major_m * big_f * self._t(phi) ** n
        theta = n * (lam - math.radians(self.central_meridian))
        x = self.false_easting + rho * np.sin(theta)
        y = self.false_northing + rho0 - rho * np.cos(theta)
        return x, y

    def inverse(self, x, y):
        """Map (x, y) in metres to longitude/latitude in degrees."""
        n, big_f, rho0 = self._constants()
        e = self.eccentricity
        dx = np.asarray(x, dtype=float) - self.false_easting
        dy = rho0 - (np.asarray(y, dtype=float) - self.false_northing)
        sign = 1.0 if n >= 0 else -1.0
        rho = sign * np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(sign * dx, sign * dy)
        t = (rho / (self.semi_major_m * big_f)) ** (1.0 / n)
        lam = theta / n + math.radians(self.central_meridian)
        # Snyder 7-9, iterated: converges in a handful of steps at any
        # latitude a chart covers.
        phi = math.pi / 2 - 2 * np.arctan(t)
        for _ in range(8):
            es = e * np.sin(phi)
            phi = math.pi / 2 - 2 * np.arctan(t * ((1 - es) / (1 + es)) ** (e / 2))
        return np.degrees(lam), np.degrees(phi)


# --- georeferencing ---------------------------------------------------------


@dataclass(frozen=True)
class Georeference:
    """Where the image sits: pixel size, top-left corner, and the projection.

    `origin` is the map coordinate of the *outer corner* of the top-left
    pixel. A world file (.tfw) states the *centre* of that pixel instead, and
    `from_world_file` shifts it by half a pixel so both mean the same thing.
    """

    pixel_width_m: float
    pixel_height_m: float
    origin_x: float
    origin_y: float
    projection: LambertConformalConic

    def to_pixel(self, lon, lat):
        """Longitude/latitude to fractional (col, row). Vectorised."""
        x, y = self.projection.forward(lon, lat)
        col = (x - self.origin_x) / self.pixel_width_m
        row = (self.origin_y - y) / self.pixel_height_m
        return col, row

    def to_lonlat(self, col, row):
        """Fractional (col, row) to longitude/latitude. Vectorised."""
        x = self.origin_x + np.asarray(col, dtype=float) * self.pixel_width_m
        y = self.origin_y - np.asarray(row, dtype=float) * self.pixel_height_m
        return self.projection.inverse(x, y)

    def lonlat_bounds(self, width: int, height: int) -> tuple[float, float, float, float]:
        """(west, south, east, north) of the image, by walking its edges.

        The corners alone are not enough: on a conic projection the top
        edge of a chart bows, so the northernmost point of a wide chart is
        mid-edge, not at a corner.
        """
        steps = 64
        cols = np.concatenate(
            [
                np.linspace(0, width, steps),
                np.full(steps, width),
                np.linspace(width, 0, steps),
                np.zeros(steps),
            ]
        )
        rows = np.concatenate(
            [
                np.zeros(steps),
                np.linspace(0, height, steps),
                np.full(steps, height),
                np.linspace(height, 0, steps),
            ]
        )
        lon, lat = self.to_lonlat(cols, rows)
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


# --- the file ---------------------------------------------------------------


@dataclass(frozen=True)
class GeoTiff:
    """What the header of one chart file says, before any pixel is decoded."""

    path: Path
    width: int
    height: int
    bits_per_sample: int
    samples_per_pixel: int
    compression: int
    photometric: int
    # 256 (r, g, b) triples, 0-255, for a palette image; None otherwise.
    palette: tuple[tuple[int, int, int], ...] | None
    # The TIFF DateTime tag, "YYYY:MM:DD HH:MM:SS", if present.
    datetime: str | None
    citation: str | None
    georeference: Georeference | None

    @property
    def is_palette(self) -> bool:
        return self.photometric == PHOTOMETRIC_PALETTE and self.palette is not None


def _read_ifd(data: bytes, byte_order: str, offset: int) -> dict[int, tuple[int, int, bytes]]:
    """Every tag in the first image file directory: tag -> (type, count, raw)."""
    (count,) = struct.unpack_from(byte_order + "H", data, offset)
    entries: dict[int, tuple[int, int, bytes]] = {}
    for i in range(count):
        base = offset + 2 + 12 * i
        tag, kind, n = struct.unpack_from(byte_order + "HHI", data, base)
        size = _TYPE_SIZE.get(kind, 1) * n
        if size <= 4:
            raw = data[base + 8 : base + 8 + size]
        else:
            (value_offset,) = struct.unpack_from(byte_order + "I", data, base + 8)
            raw = data[value_offset : value_offset + size]
        entries[tag] = (kind, n, raw)
    return entries


def _values(entry: tuple[int, int, bytes], byte_order: str) -> tuple:
    kind, n, raw = entry
    if kind == 2:
        return (raw.split(b"\0", 1)[0].decode("latin-1"),)
    fmt = _TYPE_FORMAT.get(kind)
    if fmt is None:
        return ()
    return struct.unpack(byte_order + fmt * n, raw[: _TYPE_SIZE[kind] * n])


def _geokeys(entries, byte_order: str) -> dict[int, float | int | str]:
    """The GeoKey directory, resolved: key id -> its value."""
    directory = entries.get(TAG_GEO_KEY_DIRECTORY)
    if directory is None:
        return {}
    shorts = _values(directory, byte_order)
    doubles = (
        _values(entries[TAG_GEO_DOUBLE_PARAMS], byte_order)
        if TAG_GEO_DOUBLE_PARAMS in entries
        else ()
    )
    text = (
        _values(entries[TAG_GEO_ASCII_PARAMS], byte_order)[0]
        if TAG_GEO_ASCII_PARAMS in entries
        else ""
    )
    keys: dict[int, float | int | str] = {}
    number_of_keys = shorts[3]
    for i in range(number_of_keys):
        key, location, count, value = shorts[4 + 4 * i : 8 + 4 * i]
        if location == 0:
            keys[key] = value
        elif location == TAG_GEO_DOUBLE_PARAMS:
            keys[key] = doubles[value] if count == 1 else doubles[value : value + count]
        elif location == TAG_GEO_ASCII_PARAMS:
            keys[key] = text[value : value + count].rstrip("|")
    return keys


def _projection(keys: dict) -> LambertConformalConic | None:
    if keys.get(KEY_MODEL_TYPE) != MODEL_TYPE_PROJECTED:
        return None
    transform = keys.get(KEY_PROJ_COORD_TRANS)
    if transform not in (COORD_TRANS_LCC_2SP, COORD_TRANS_LCC_1SP):
        raise TiffError(f"unsupported projection (GeoTIFF coordinate transformation {transform})")
    if keys.get(KEY_PROJ_LINEAR_UNITS, LINEAR_UNIT_METRE) != LINEAR_UNIT_METRE:
        raise TiffError("projected units are not metres")
    # 2SP charts carry their origin as a "false origin", 1SP as a "natural
    # origin"; the FAA writes 2SP. Accept either spelling.
    parallel_1 = keys.get(KEY_STD_PARALLEL_1, keys.get(KEY_NAT_ORIGIN_LAT))
    parallel_2 = keys.get(KEY_STD_PARALLEL_2, parallel_1)
    central = keys.get(KEY_FALSE_ORIGIN_LONG, keys.get(KEY_NAT_ORIGIN_LONG))
    origin_lat = keys.get(KEY_FALSE_ORIGIN_LAT, keys.get(KEY_NAT_ORIGIN_LAT))
    if parallel_1 is None or central is None or origin_lat is None:
        raise TiffError("Lambert projection is missing its parallels or origin")
    return LambertConformalConic(
        standard_parallel_1=float(parallel_1),
        standard_parallel_2=float(parallel_2),
        central_meridian=float(central),
        latitude_of_origin=float(origin_lat),
        false_easting=float(keys.get(KEY_FALSE_ORIGIN_EASTING, keys.get(KEY_FALSE_EASTING, 0.0))),
        false_northing=float(
            keys.get(KEY_FALSE_ORIGIN_NORTHING, keys.get(KEY_FALSE_NORTHING, 0.0))
        ),
        semi_major_m=float(keys.get(KEY_SEMI_MAJOR_AXIS, DEFAULT_SEMI_MAJOR_M)),
        inverse_flattening=float(keys.get(KEY_INV_FLATTENING, DEFAULT_INV_FLATTENING)),
    )


def _georeference(entries, byte_order: str, keys: dict) -> Georeference | None:
    projection = _projection(keys)
    if projection is None:
        return None
    if TAG_MODEL_PIXEL_SCALE not in entries or TAG_MODEL_TIEPOINT not in entries:
        if TAG_MODEL_TRANSFORMATION in entries:
            raise TiffError("ModelTransformation matrices are not supported; need scale+tiepoint")
        return None
    scale_x, scale_y = _values(entries[TAG_MODEL_PIXEL_SCALE], byte_order)[:2]
    tie = _values(entries[TAG_MODEL_TIEPOINT], byte_order)
    col, row, _, x, y, _ = tie[:6]
    # Whatever pixel is tied down, shift to the corner of pixel (0, 0).
    origin_x = x - col * scale_x
    origin_y = y + row * scale_y
    if keys.get(KEY_RASTER_TYPE) == RASTER_PIXEL_IS_POINT:
        origin_x -= scale_x / 2
        origin_y += scale_y / 2
    return Georeference(
        float(scale_x), float(scale_y), float(origin_x), float(origin_y), projection
    )


def read_header(path: Path) -> GeoTiff:
    """Parse the first IFD of a TIFF. Reads the header bytes only."""
    path = Path(path)
    with path.open("rb") as handle:
        head = handle.read(8)
        if len(head) < 8:
            raise TiffError(f"{path.name}: not a TIFF")
        if head[:2] == b"II":
            byte_order = "<"
        elif head[:2] == b"MM":
            byte_order = ">"
        else:
            raise TiffError(f"{path.name}: not a TIFF")
        (magic,) = struct.unpack_from(byte_order + "H", head, 2)
        if magic == 43:
            raise TiffError(f"{path.name}: BigTIFF is not supported")
        if magic != 42:
            raise TiffError(f"{path.name}: not a TIFF")
        (first_ifd,) = struct.unpack_from(byte_order + "I", head, 4)
        # Tag values can point anywhere, so keep the whole file mapped rather
        # than seeking; strip offsets are at the front and reading the header
        # this way costs one read of a few hundred kilobytes at most.
        handle.seek(0)
        data = handle.read(first_ifd + 2 + 12 * 4096)
    entries = _read_ifd(data, byte_order, first_ifd)

    # A value table (colour map, strip offsets) can live past what was read.
    needed = max(
        (
            struct.unpack_from(byte_order + "I", data, first_ifd + 2 + 12 * i + 8)[0]
            + _TYPE_SIZE.get(kind, 1) * n
            for i, (kind, n, _) in enumerate(entries.values())
            if _TYPE_SIZE.get(kind, 1) * n > 4
        ),
        default=0,
    )
    if needed > len(data):
        with path.open("rb") as handle:
            data = handle.read(needed)
        entries = _read_ifd(data, byte_order, first_ifd)

    def one(tag: int, default=None):
        if tag not in entries:
            return default
        values = _values(entries[tag], byte_order)
        return values[0] if values else default

    width = one(TAG_IMAGE_WIDTH)
    height = one(TAG_IMAGE_LENGTH)
    if not width or not height:
        raise TiffError(f"{path.name}: no image size")

    palette = None
    if TAG_COLOR_MAP in entries:
        table = _values(entries[TAG_COLOR_MAP], byte_order)
        size = len(table) // 3
        palette = tuple(
            (table[i] >> 8, table[size + i] >> 8, table[2 * size + i] >> 8) for i in range(size)
        )

    keys = _geokeys(entries, byte_order)
    return GeoTiff(
        path=path,
        width=int(width),
        height=int(height),
        bits_per_sample=int(one(TAG_BITS_PER_SAMPLE, 8)),
        samples_per_pixel=int(one(TAG_SAMPLES_PER_PIXEL, 1)),
        compression=int(one(TAG_COMPRESSION, 1)),
        photometric=int(one(TAG_PHOTOMETRIC, 0)),
        palette=palette,
        datetime=one(TAG_DATETIME),
        citation=keys.get(KEY_CITATION) or None,
        georeference=_georeference(entries, byte_order, keys),
    )


def read_world_file(path: Path, projection: LambertConformalConic) -> Georeference:
    """A six-line ESRI world file (.tfw): the same placement, minus the projection.

    Used only when a TIFF has lost its own tags; the projection has to come
    from somewhere else, since a world file does not name one.
    """
    lines = [float(line) for line in Path(path).read_text().split()]
    if len(lines) < 6:
        raise TiffError(f"{Path(path).name}: not a world file")
    scale_x, rot_y, rot_x, scale_y, centre_x, centre_y = lines[:6]
    if rot_x or rot_y:
        raise TiffError(f"{Path(path).name}: rotated world files are not supported")
    # A world file gives the centre of the top-left pixel.
    return Georeference(
        pixel_width_m=scale_x,
        pixel_height_m=-scale_y,
        origin_x=centre_x - scale_x / 2,
        origin_y=centre_y - scale_y / 2,
        projection=projection,
    )


def read_pixels(path: Path) -> np.ndarray:
    """Decode a palette TIFF to its index array, shape (height, width), uint8.

    Pillow's decompression-bomb guard is sized for photographs; a sectional
    is two hundred megapixels and entirely expected.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        if image.mode != "P":
            raise TiffError(f"{Path(path).name}: expected a palette image, got mode {image.mode}")
        return np.asarray(image, dtype=np.uint8)
