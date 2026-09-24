"""Shared fixtures.

`geotiff` writes a small palette GeoTIFF with the same tags an FAA chart
carries, so the reader, loader and renderer can be tested without the
hundred-megabyte real thing on disk. It is placed at the projection origin
of the San Francisco TAC, with one kilometre pixels, so its footprint is a
few tiles at zoom 8 and the numbers are easy to reason about.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Same projection as the San Francisco TAC / Flyway charts.
SF_TAC_PROJECTION = {
    "latitude_of_origin": 37.6,
    "central_meridian": -122.25,
    "standard_parallel_1": 45.0,
    "standard_parallel_2": 33.0,
}

FGDC_HTML = """<html><body>
<dl><dt><em>Beginning_Date:</em>  {begins}</dt>
<dt><em>Ending_Date:</em>  {ends}</dt>
<dt><em>Publication_Date:</em>  {published}</dt>
<dt><em>Row_Count:</em>  {rows}</dt>
<dt><em>Column_Count:</em>  {columns}</dt></dl>
</body></html>
"""


def write_geotiff(
    path: Path,
    *,
    width: int = 64,
    height: int = 48,
    pixel_m: float = 1000.0,
    origin: tuple[float, float] | None = None,
    projection: dict | None = None,
    pixel_is_point: bool = False,
) -> Path:
    """A palette GeoTIFF: left half palette index 1 (red), right half 2 (blue)."""
    from PIL import Image, TiffImagePlugin

    projection = {**SF_TAC_PROJECTION, **(projection or {})}
    if origin is None:
        # Centred on the projection origin (map 0, 0).
        origin = (-width * pixel_m / 2, height * pixel_m / 2)

    image = Image.new("P", (width, height), 0)
    palette = [255, 255, 255, 255, 0, 0, 0, 0, 255] + [0, 0, 0] * 253
    image.putpalette(palette)
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), 1 if x < width // 2 else 2)

    doubles = (
        projection["latitude_of_origin"],
        projection["central_meridian"],
        projection["standard_parallel_1"],
        projection["standard_parallel_2"],
        0.0,
        0.0,
        298.257222101,
        6378137.0,
    )
    keys = [
        (1024, 0, 1, 1),  # projected
        (1025, 0, 1, 2 if pixel_is_point else 1),
        (1026, 34737, 24, 0),
        (2048, 0, 1, 4269),  # NAD83
        (2057, 34736, 1, 7),
        (2059, 34736, 1, 6),
        (3072, 0, 1, 32767),
        (3075, 0, 1, 8),  # LCC 2SP
        (3076, 0, 1, 9001),
        (3078, 34736, 1, 2),
        (3079, 34736, 1, 3),
        (3084, 34736, 1, 1),
        (3085, 34736, 1, 0),
        (3086, 34736, 1, 4),
        (3087, 34736, 1, 5),
    ]
    directory = (1, 1, 0, len(keys)) + tuple(v for key in keys for v in key)

    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[33550] = (pixel_m, pixel_m, 0.0)
    info.tagtype[33550] = 12
    tie_x, tie_y = origin
    if pixel_is_point:
        tie_x += pixel_m / 2
        tie_y -= pixel_m / 2
    info[33922] = (0.0, 0.0, 0.0, tie_x, tie_y, 0.0)
    info.tagtype[33922] = 12
    info[34735] = directory
    info.tagtype[34735] = 3
    info[34736] = doubles
    info.tagtype[34736] = 12
    info[34737] = "Lambert Conformal Conic|NAD83|"
    info.tagtype[34737] = 2
    info[306] = "2026:08:10 10:40:40"
    info.tagtype[306] = 2
    image.save(path, format="TIFF", tiffinfo=info, compression="tiff_lzw")
    return path


def write_fgdc(path: Path, *, begins="20260903", ends="20261028", rows=48, columns=64) -> Path:
    path.write_text(
        FGDC_HTML.format(begins=begins, ends=ends, published=begins, rows=rows, columns=columns)
    )
    return path


@pytest.fixture
def geotiff(tmp_path):
    """Writer for synthetic charts under a temporary data/charts tree."""

    def make(kind: str = "tac", name: str = "Test TAC", *, with_metadata=True, **options):
        folder = tmp_path / "charts" / kind / name.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        path = write_geotiff(folder / f"{name}.tif", **options)
        if with_metadata:
            write_fgdc(
                folder / f"{name}.htm",
                rows=options.get("height", 48),
                columns=options.get("width", 64),
            )
        return path

    make.root = tmp_path / "charts"
    return make
