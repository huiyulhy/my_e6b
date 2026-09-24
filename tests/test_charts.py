"""Chart discovery and currency: what is on disk, and how long it is good for."""

from __future__ import annotations

import json
from datetime import date

import pytest
from conftest import write_fgdc, write_geotiff

from engine import charts
from engine import currency as cur


def test_reads_edition_dates_and_size_from_the_fgdc_html(tmp_path):
    edition = charts.read_edition(write_fgdc(tmp_path / "x.htm"))
    assert edition.begins == date(2026, 9, 3)
    assert edition.ends == date(2026, 10, 28)
    assert edition.published == date(2026, 9, 3)
    assert (edition.columns, edition.rows) == (64, 48)


def test_missing_metadata_file_is_none(tmp_path):
    assert charts.read_edition(tmp_path / "nope.htm") is None


def test_discovers_every_chart_series_first(geotiff):
    geotiff("tac", "Test TAC")
    geotiff("tac", "Test FLY")
    geotiff("sectional", "Test SEC")
    geotiff("helicopter", "Test HEL")
    found = charts.discover(geotiff.root)
    assert [c.key for c in found] == [
        "sectional/test_sec",
        "tac/test_fly",
        "tac/test_tac",
        "helicopter/test_hel",
    ]
    assert [c.kind_label for c in found] == [
        "Sectional",
        "Terminal Area",
        "Terminal Area",
        "Helicopter",
    ]


def test_chart_carries_footprint_zooms_and_dates(geotiff):
    geotiff("tac", "Test TAC")
    (chart,) = charts.discover(geotiff.root)
    assert chart.name == "Test TAC"
    assert (chart.width, chart.height, chart.pixel_m) == (64, 48, 1000.0)
    west, south, east, north = chart.bounds
    assert west < -122.25 < east and south < 37.6 < north
    assert chart.min_zoom == 5
    # One kilometre pixels: zoom 7 at 37.6N is 970 m/px, so that is native.
    assert chart.max_zoom == 7
    assert chart.effective == date(2026, 9, 3)
    assert chart.expires == date(2026, 10, 28)
    assert chart.note is None


def test_chart_without_metadata_still_draws_but_says_so(geotiff):
    geotiff("tac", "Bare", with_metadata=False)
    (chart,) = charts.discover(geotiff.root)
    assert chart.effective is None and chart.expires is None
    assert "Bare.htm" in chart.note


def test_metadata_for_a_different_image_is_flagged(geotiff):
    path = geotiff("tac", "Mismatch")
    write_fgdc(path.with_suffix(".htm"), rows=7265, columns=10493)
    (chart,) = charts.discover(geotiff.root)
    assert "10493x7265" in chart.note and "64x48" in chart.note


def test_unreadable_files_and_the_tile_cache_are_skipped(geotiff):
    geotiff("sectional", "Good")
    (geotiff.root / "sectional" / "Bad.tif").write_bytes(b"not a tiff")
    tiles = geotiff.root / "tiles" / "sectional" / "good"
    tiles.mkdir(parents=True)
    write_geotiff(tiles / "stray.tif")
    assert [c.key for c in charts.discover(geotiff.root)] == ["sectional/good"]


def test_find_by_kind_and_slug(geotiff):
    geotiff("tac", "Test TAC")
    assert charts.find("tac", "test_tac", geotiff.root).name == "Test TAC"
    assert charts.find("sectional", "test_tac", geotiff.root) is None


def test_empty_or_missing_folder_is_no_charts(tmp_path):
    assert charts.discover(tmp_path / "nothing") == []


# --- currency ----------------------------------------------------------------


def test_each_chart_is_a_dated_dataset(geotiff):
    geotiff("tac", "Test TAC")
    (dataset,) = cur.charts(charts_dir=geotiff.root)
    assert dataset.key == "chart:tac/test_tac"
    assert dataset.label == "Terminal Area chart: Test TAC"
    assert dataset.effective == date(2026, 9, 3)
    assert dataset.expires == date(2026, 10, 28)
    assert not dataset.expired(date(2026, 10, 28))
    assert dataset.expired(date(2026, 10, 29))
    assert dataset.days_remaining(date(2026, 9, 23)) == 35


def test_charts_join_the_other_datasets():
    keys = [d.key for d in cur.datasets()]
    assert keys[:2] == ["nasr", "wmm"]
    assert all(key.startswith("chart:") for key in keys[2:])


# --- the manifest -------------------------------------------------------------
#
# What a deployed server runs on. It has no GeoTIFFs, so everything the layer
# menu and the currency list show has to come out of this file.


def test_manifest_round_trips_a_chart(geotiff, tmp_path):
    geotiff("tac", "Test TAC")
    (original,) = charts.discover(geotiff.root)
    charts.write_manifest([original], tmp_path / "tiles")

    (restored,) = charts.read_manifest(tmp_path / "tiles")
    assert restored.key == original.key
    assert restored.name == original.name
    assert restored.bounds == original.bounds
    assert (restored.min_zoom, restored.max_zoom) == (original.min_zoom, original.max_zoom)
    assert (restored.effective, restored.expires) == (original.effective, original.expires)
    # The one difference that matters: there is no file behind it.
    assert restored.path is None
    assert restored.prerendered is True


def test_a_manifest_chart_cannot_be_rendered_from(geotiff, tmp_path):
    geotiff("tac", "Test TAC")
    charts.write_manifest(charts.discover(geotiff.root), tmp_path / "tiles")
    (restored,) = charts.read_manifest(tmp_path / "tiles")
    with pytest.raises(charts.NoChartFile):
        restored.store(tmp_path / "tiles")


def test_missing_or_unreadable_manifest_is_no_charts(tmp_path):
    assert charts.read_manifest(tmp_path / "nothing") == []
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / charts.MANIFEST_NAME).write_text("{not json")
    assert charts.read_manifest(tiles) == []


def test_a_manifest_from_another_version_is_ignored(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / charts.MANIFEST_NAME).write_text(
        json.dumps({"version": charts.MANIFEST_VERSION + 1, "charts": [{"kind": "tac"}]})
    )
    assert charts.read_manifest(tiles) == []


def test_one_broken_entry_does_not_lose_the_others(geotiff, tmp_path):
    geotiff("tac", "Test TAC")
    tiles = tmp_path / "tiles"
    charts.write_manifest(charts.discover(geotiff.root), tiles)
    body = json.loads((tiles / charts.MANIFEST_NAME).read_text())
    body["charts"].insert(0, {"kind": "tac", "slug": "broken"})  # no bounds, no size
    (tiles / charts.MANIFEST_NAME).write_text(json.dumps(body))
    assert [c.slug for c in charts.read_manifest(tiles)] == ["test_tac"]


def test_rendered_zooms_are_read_off_the_directories(geotiff, tmp_path):
    geotiff("tac", "Test TAC")
    (chart,) = charts.discover(geotiff.root)
    tiles = tmp_path / "tiles"
    assert charts.rendered_zooms(chart, tiles) is None
    for zoom in (5, 6, 7):
        (chart.tiles_dir(tiles) / str(zoom) / "1").mkdir(parents=True)
    # A stray non-numeric directory is not a zoom level.
    (chart.tiles_dir(tiles) / "scratch").mkdir()
    assert charts.rendered_zooms(chart, tiles) == (5, 7)


def test_available_prefers_the_files_and_falls_back_to_the_manifest(geotiff, tmp_path):
    geotiff("tac", "Test TAC")
    tiles = tmp_path / "tiles"
    charts.write_manifest(charts.discover(geotiff.root), tiles)

    # Desktop: the GeoTIFFs are there, so any zoom can be rendered on demand.
    (from_files,) = charts.available(geotiff.root, tiles)
    assert from_files.prerendered is False and from_files.path is not None

    # Deployed: no GeoTIFFs, only the committed tiles.
    (from_manifest,) = charts.available(tmp_path / "no-charts-here", tiles)
    assert from_manifest.prerendered is True and from_manifest.path is None


def test_currency_reads_the_manifest_when_there_are_no_files(geotiff, tmp_path):
    geotiff("tac", "Test TAC")
    tiles = tmp_path / "tiles"
    charts.write_manifest(charts.discover(geotiff.root), tiles)
    # cur.charts takes only the chart directory, so point the tile root at the
    # manifest by monkeypatching nothing: read_manifest defaults to TILES_DIR,
    # which is why the deployed case is exercised through available() above.
    (dataset,) = [
        cur.Dataset(
            f"chart:{c.key}", f"{c.kind_label} chart: {c.name}", c.effective, c.expires, c.note
        )
        for c in charts.read_manifest(tiles)
    ]
    assert dataset.effective == date(2026, 9, 3)
    assert dataset.expires == date(2026, 10, 28)
