"""The one decision the chart endpoints make: where tiles are fetched from.

Everything else about a chart is settled in `engine/charts.py`. This is the
seam between the two run modes -- rendered on demand on the desktop, static
files on a deployed instance -- so it is tested directly.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import write_fgdc, write_geotiff

from engine import charts as ch
from server.main import _chart_json, _tiles_url


@pytest.fixture
def chart(tmp_path):
    folder = tmp_path / "charts" / "tac" / "Test_TAC"
    folder.mkdir(parents=True)
    write_geotiff(folder / "Test TAC.tif")
    write_fgdc(folder / "Test TAC.htm")
    (found,) = ch.discover(tmp_path / "charts")
    return found


def test_a_chart_on_disk_is_rendered_on_demand(chart):
    assert chart.prerendered is False
    assert _tiles_url(chart) == "/api/charts/tiles/tac/test_tac/{z}/{x}/{y}.png"


def test_a_prerendered_chart_is_served_as_static_files(chart, tmp_path):
    ch.write_manifest([chart], tmp_path / "tiles")
    (deployed,) = ch.read_manifest(tmp_path / "tiles")
    assert deployed.prerendered is True
    # Under /data, which is mounted as static files, so no Python runs per
    # tile and the GeoTIFF never has to be on the machine.
    assert _tiles_url(deployed) == "/data/charts/tiles/tac/test_tac/{z}/{x}/{y}.png"


def test_the_json_carries_what_the_layer_menu_draws(chart):
    body = _chart_json(chart, date(2026, 9, 23))
    assert body["kind_label"] == "Terminal Area"
    assert body["name"] == "Test TAC"
    assert body["bounds"] == list(chart.bounds)
    assert (body["min_zoom"], body["max_zoom"]) == (chart.min_zoom, chart.max_zoom)
    assert body["expires"] == "2026-10-28"
    assert body["expired"] is False
    assert body["days_remaining"] == 35
    assert body["prerendered"] is False


def test_an_expired_chart_says_so(chart):
    body = _chart_json(chart, date(2026, 11, 1))
    assert body["expired"] is True
    assert body["days_remaining"] == -4
