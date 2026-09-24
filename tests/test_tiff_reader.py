"""The GeoTIFF reader: tags, georeferencing, and the Lambert projection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import SF_TAC_PROJECTION, write_geotiff

from engine import tiff_reader as tr

REAL_SECTIONAL = Path("data/charts/sectional/San_Francisco/San Francisco SEC.tif")


@pytest.fixture
def chart(tmp_path):
    return write_geotiff(tmp_path / "Test TAC.tif")


def test_reads_size_palette_and_tags(chart):
    header = tr.read_header(chart)
    assert (header.width, header.height) == (64, 48)
    assert header.compression == 5  # LZW, as the FAA ships them
    assert header.is_palette
    assert header.palette[:3] == ((255, 255, 255), (255, 0, 0), (0, 0, 255))
    assert header.datetime == "2026:08:10 10:40:40"
    assert header.citation == "Lambert Conformal Conic"


def test_georeference_comes_from_the_geotiff_tags(chart):
    geo = tr.read_header(chart).georeference
    assert geo is not None
    assert geo.pixel_width_m == geo.pixel_height_m == 1000.0
    # The image is centred on the projection origin: 64 km wide, 48 km tall.
    assert (geo.origin_x, geo.origin_y) == (-32000.0, 24000.0)
    assert geo.projection == tr.LambertConformalConic(**SF_TAC_PROJECTION)


def test_pixel_is_point_shifts_the_tiepoint_to_the_corner(tmp_path):
    corner = tr.read_header(write_geotiff(tmp_path / "area.tif")).georeference
    point = tr.read_header(write_geotiff(tmp_path / "point.tif", pixel_is_point=True)).georeference
    assert (point.origin_x, point.origin_y) == (corner.origin_x, corner.origin_y)


def test_world_file_states_the_pixel_centre(tmp_path, chart):
    geo = tr.read_header(chart).georeference
    tfw = tmp_path / "Test TAC.tfw"
    tfw.write_text("1000.0\n0.0\n0.0\n-1000.0\n-31500.0\n23500.0\n")
    from_world = tr.read_world_file(tfw, geo.projection)
    assert (from_world.origin_x, from_world.origin_y) == (geo.origin_x, geo.origin_y)


def test_pixels_decode_to_the_palette_indices(chart):
    pixels = tr.read_pixels(chart)
    assert pixels.shape == (48, 64) and pixels.dtype == np.uint8
    assert pixels[:, :32].max() == 1 and pixels[:, 32:].min() == 2


def test_not_a_tiff_is_refused(tmp_path):
    bogus = tmp_path / "chart.tif"
    bogus.write_bytes(b"PNG\r\n" * 10)
    with pytest.raises(tr.TiffError):
        tr.read_header(bogus)


# --- the projection ----------------------------------------------------------


def test_origin_maps_to_false_origin():
    lcc = tr.LambertConformalConic(**SF_TAC_PROJECTION)
    x, y = lcc.forward(-122.25, 37.6)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6


def test_forward_and_inverse_round_trip_over_the_bay_area():
    lcc = tr.LambertConformalConic(**SF_TAC_PROJECTION)
    lon = np.array([-123.9, -122.25, -121.4, -121.36])
    lat = np.array([36.86, 37.6, 38.26, 36.9])
    back_lon, back_lat = lcc.inverse(*lcc.forward(lon, lat))
    assert np.allclose(back_lon, lon, atol=1e-9)
    assert np.allclose(back_lat, lat, atol=1e-9)


def test_projection_is_conformal_to_snyders_worked_example():
    # Snyder (1987) p. 296: Clarke 1866, parallels 33 and 45, origin 23N 96W,
    # point 35N 75W -> x = 1 894 410.9, y = 1 564 649.5 m.
    lcc = tr.LambertConformalConic(
        standard_parallel_1=33.0,
        standard_parallel_2=45.0,
        central_meridian=-96.0,
        latitude_of_origin=23.0,
        semi_major_m=6378206.4,
        inverse_flattening=294.9786982,
    )
    x, y = lcc.forward(-75.0, 35.0)
    assert float(x) == pytest.approx(1894410.9, abs=0.5)
    assert float(y) == pytest.approx(1564649.5, abs=0.5)


def test_bounds_walk_the_edges_not_just_the_corners(chart):
    header = tr.read_header(chart)
    west, south, east, north = header.georeference.lonlat_bounds(header.width, header.height)
    # 64 km at 37.6N is about 0.72 degrees of longitude, 48 km about 0.43 of latitude.
    assert west == pytest.approx(-122.25 - 0.36, abs=0.01)
    assert east == pytest.approx(-122.25 + 0.36, abs=0.01)
    assert south == pytest.approx(37.6 - 0.216, abs=0.01)
    assert north == pytest.approx(37.6 + 0.216, abs=0.01)


@pytest.mark.skipif(not REAL_SECTIONAL.exists(), reason="no sectional downloaded")
def test_real_sectional_footprint_matches_faa_metadata():
    # The bounding coordinates printed in San Francisco SEC.htm.
    header = tr.read_header(REAL_SECTIONAL)
    assert (header.width, header.height) == (16658, 12340)
    west, south, east, north = header.georeference.lonlat_bounds(header.width, header.height)
    assert west == pytest.approx(-125.946499, abs=1e-4)
    assert east == pytest.approx(-117.633281, abs=1e-4)
    assert south == pytest.approx(35.857057, abs=1e-4)
    assert north == pytest.approx(40.639460, abs=1e-4)
