"""Reprojecting a chart into Web Mercator tiles."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import write_geotiff

from engine import chart_render as cr

RED = (255, 0, 0)
BLUE = (0, 0, 255)


@pytest.fixture
def raster(tmp_path):
    return cr.ChartRaster.load(write_geotiff(tmp_path / "Test TAC.tif"))


# --- tile arithmetic ----------------------------------------------------------


def test_tile_scheme_is_xyz():
    assert cr.tile_at(-180, 85, 0) == (0, 0)
    assert cr.tile_at(0, 0, 1) == (1, 1)
    assert cr.tile_at(-122.25, 37.6, 8) == (41, 99)


def test_tile_range_is_inclusive_and_north_first():
    x0, y0, x1, y1 = cr.tile_range((-122.6, 37.4, -121.9, 37.8), 8)
    assert (x0, x1) == (40, 41)
    assert (y0, y1) == (98, 99)


def test_native_zoom_of_the_real_charts():
    # 42 m sectional pixels at 38N: zoom 12 is 30 m, zoom 11 is 60 m.
    assert cr.native_zoom(42.34, 38.2) == 12
    # 21 m TAC pixels: one more.
    assert cr.native_zoom(21.17, 37.6) == 13


def test_tile_pixel_centres_span_the_tile():
    lon, lat = cr.tile_pixel_lonlat(1, 0, 0)
    assert lon.shape == lat.shape == (256, 256)
    assert lon[0, 0] == pytest.approx(-180 + 180 / 512)
    assert lon[0, -1] == pytest.approx(-180 / 512)
    assert lat[0, 0] > 85 and 0 < lat[-1, 0] < 0.5


# --- rendering -------------------------------------------------------------------


def test_tile_over_the_chart_is_opaque_and_the_right_way_round(raster):
    # A zoom 11 tile is 15 km across; the chart extends 24 km each way.
    x, y = cr.tile_at(-122.25, 37.6, 11)
    tile = raster.render_tile(11, x, y)
    assert tile.shape == (256, 256, 4)
    assert tile[..., 3].min() == 255
    # The chart is red west of the central meridian and blue east of it. A
    # pixel astride the seam can go either way, so leave the seam out.
    lon, _ = cr.tile_pixel_lonlat(11, x, y)
    west = lon[0] < -122.25 - 0.002
    east = lon[0] > -122.25 + 0.002
    assert west.any() and east.any()
    assert (tile[128, west, :3] == RED).all()
    assert (tile[128, east, :3] == BLUE).all()


def test_tile_off_the_chart_is_none(raster):
    assert raster.render_tile(9, *cr.tile_at(-100.0, 40.0, 9)) is None


def test_tile_on_the_edge_is_transparent_outside(raster):
    west, _, _, north = raster.georeference.lonlat_bounds(64, 48)
    tile = raster.render_tile(8, *cr.tile_at(west, north, 8))
    alpha = tile[..., 3]
    assert alpha.min() == 0 and alpha.max() == 255
    # Wherever the chart is drawn it is red or blue, never the palette's zero.
    drawn = tile[alpha == 255][:, :3]
    assert set(map(tuple, np.unique(drawn, axis=0))) <= {RED, BLUE}


def test_zoomed_out_tiles_sample_the_pyramid(raster):
    assert raster.max_level == 0  # 64 px wide: too small to halve
    assert raster.level_for(7, 37.6) == 0
    # 32 copies each way: 2048 x 1536, enough to halve once but not twice.
    big = cr.ChartRaster(raster.header, np.tile(raster.indices, (32, 32)), raster.palette)
    assert big.max_level == 1
    assert big.level_for(5, 37.6) == 1
    halved = big.level(1)
    assert halved.shape == (768, 1024, 3)
    # Box filtering keeps solid colour solid; only the seams blend.
    assert (halved[0, 0] == RED).all() and (halved[0, -1] == BLUE).all()


def test_png_round_trip(raster):
    from io import BytesIO

    from PIL import Image

    tile = raster.render_tile(9, *cr.tile_at(-122.25, 37.6, 9))
    with Image.open(BytesIO(cr.png_bytes(tile))) as image:
        assert image.mode == "P" and image.size == (256, 256)
        back = np.asarray(image.convert("RGBA"))
    assert np.array_equal(back[..., 3], tile[..., 3])
    drawn = tile[..., 3] == 255
    assert np.array_equal(back[drawn][:, :3], tile[drawn][:, :3])


# --- the on-disk store --------------------------------------------------------------


def test_store_renders_once_and_then_reads_the_file(tmp_path):
    store = cr.TileStore(write_geotiff(tmp_path / "Test TAC.tif"), tmp_path / "tiles")
    z, (x, y) = 9, cr.tile_at(-122.25, 37.6, 9)
    first = store.get(z, x, y)
    path = store.tile_path(z, x, y)
    assert path.read_bytes() == first
    # Replace the cached file: the second call must come from disk.
    path.write_bytes(b"cached")
    assert store.get(z, x, y) == b"cached"


def test_store_answers_empty_tiles_with_a_transparent_png(tmp_path):
    store = cr.TileStore(write_geotiff(tmp_path / "Test TAC.tif"), tmp_path / "tiles")
    assert store.get(9, *cr.tile_at(-100.0, 40.0, 9)) == cr.EMPTY_TILE_PNG


def test_render_all_covers_the_footprint(tmp_path):
    chart = write_geotiff(tmp_path / "Test TAC.tif")
    store = cr.TileStore(chart, tmp_path / "tiles")
    bounds = store.raster.georeference.lonlat_bounds(64, 48)
    count = store.render_all(bounds, 7, 8)
    assert count == len(list((tmp_path / "tiles").rglob("*.png")))
    assert (tmp_path / "tiles" / "8").is_dir()


# --- palette encoding ----------------------------------------------------------------
#
# Tiles are committed and served as static files, so their size is the size
# of the repo and of every deploy. Palette PNG is a third of RGBA here
# because the charts are palette images to begin with.


def test_tile_at_native_zoom_is_an_exact_palette_png(raster):
    from io import BytesIO

    from PIL import Image

    tile = raster.render_tile(9, *cr.tile_at(-122.25, 37.6, 9))
    with Image.open(BytesIO(cr.png_bytes(tile))) as image:
        assert image.mode == "P"
        assert "transparency" in image.info
        back = np.asarray(image.convert("RGBA"))
    # Nothing is lost: a tile sampled at the chart's own scale holds only
    # palette colours, so the encoder keeps every one of them. Only the drawn
    # pixels are compared -- what lies under a transparent one is undefined,
    # and PNG does not promise to carry it back.
    drawn = tile[..., 3] == 255
    assert np.array_equal(back[drawn][:, :3], tile[drawn][:, :3])


def test_transparency_survives_the_round_trip(raster):
    from io import BytesIO

    from PIL import Image

    west, _, _, north = raster.georeference.lonlat_bounds(64, 48)
    tile = raster.render_tile(8, *cr.tile_at(west, north, 8))
    assert (tile[..., 3] == 0).any()
    with Image.open(BytesIO(cr.png_bytes(tile))) as image:
        back = np.asarray(image.convert("RGBA"))
    assert np.array_equal(back[..., 3], tile[..., 3])


def test_a_tile_of_many_colours_is_quantised_not_refused():
    # A pyramid level holds blended colours, thousands of them. That is the
    # one place the encoder is lossy, and it must still produce a tile.
    from io import BytesIO

    from PIL import Image

    rng = np.random.default_rng(0)
    noisy = np.empty((cr.TILE_SIZE, cr.TILE_SIZE, 4), dtype=np.uint8)
    noisy[..., :3] = rng.integers(0, 256, (cr.TILE_SIZE, cr.TILE_SIZE, 3), dtype=np.uint8)
    noisy[..., 3] = 255
    with Image.open(BytesIO(cr.png_bytes(noisy))) as image:
        assert image.mode == "P"
        assert np.asarray(image).max() < 255  # room left for the transparent index


def test_palette_tiles_are_much_smaller_than_rgba(raster):
    from io import BytesIO

    from PIL import Image

    tile = raster.render_tile(9, *cr.tile_at(-122.25, 37.6, 9))
    buffer = BytesIO()
    Image.fromarray(tile, "RGBA").save(buffer, format="PNG", compress_level=6)
    assert len(cr.png_bytes(tile)) < len(buffer.getvalue())
