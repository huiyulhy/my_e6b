"""Magnetic model tests, checked against NOAA's own published test values.
from WMM_2025_TestValues.txt
"""

from pathlib import Path

import pytest

from engine import magnetic as mag

TEST_VALUES = (
    Path(__file__).resolve().parent.parent / "data" / "magnetic" / "WMM2025_TestValues.txt"
)

def load_reference() -> list[dict[str, float]]:
    """Parse NOAA's reference file. Columns are documented in its header."""
    points = []
    for line in TEST_VALUES.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = [float(v) for v in line.split()]
        points.append(
            {
                "year": f[0],
                "altitude_km": f[1],
                "lat": f[2],
                "lon": f[3],
                "declination": f[4],
                "inclination": f[5],
                "horizontal": f[6],
                "north": f[7],
                "east": f[8],
                "down": f[9],
                "total": f[10],
            }
        )
    return points


REFERENCE = load_reference()
KM_TO_FT = 1.0 / 0.0003048


class TestAgainstNOAATestValues:
    """Every one of NOAA's 100 published points must reproduce.
    Tolerances are set by NOAA's own printing precision,:
    they publish declination and inclination to two decimal places, so half a
    unit in the last place -- 0.005 deg
    Field components are published to six decimals
    """

    def test_reference_file_loaded(self):
        assert len(REFERENCE) == 100

    @pytest.mark.parametrize("point", REFERENCE, ids=lambda p: f"{p['lat']},{p['lon']}")
    def test_point(self, point):
        result = mag.field(
            point["lat"],
            point["lon"],
            altitude_ft=point["altitude_km"] * KM_TO_FT,
            decimal_year=point["year"],
        )
        # Declination wraps, so compare the signed angular difference.
        delta = (result.declination_deg - point["declination"] + 180.0) % 360.0 - 180.0
        assert abs(delta) <= 0.005, "declination"
        assert result.inclination_deg == pytest.approx(point["inclination"], abs=0.005)
        assert result.north_nt == pytest.approx(point["north"], abs=0.01)
        assert result.east_nt == pytest.approx(point["east"], abs=0.01)
        assert result.down_nt == pytest.approx(point["down"], abs=0.01)
        assert result.horizontal_nt == pytest.approx(point["horizontal"], abs=0.01)
        assert result.total_nt == pytest.approx(point["total"], abs=0.01)

class TestModelValidity:
    def test_coefficients_are_wmm2025(self):
        model = mag._model()
        assert model.epoch == pytest.approx(2025.0)
        assert model.label.startswith("WMM")

    def test_date_before_epoch_refuses(self):
        with pytest.raises(mag.OutsideModelValidity):
            mag.variation(37.0, -122.0, decimal_year=2024.0)

    def test_date_after_window_refuses(self):
        with pytest.raises(mag.OutsideModelValidity):
            mag.variation(37.0, -122.0, decimal_year=2031.0)

    def test_extrapolation_is_never_offered(self):
        """There is no escape hatch: an expired model refuses, full stop."""
        with pytest.raises(mag.OutsideModelValidity, match="expired"):
            mag.field(37.0, -122.0, decimal_year=2031.0)

    def test_window_edges_are_accepted(self):
        start, end = mag.validity_window()
        assert start == pytest.approx(2025.0)
        assert end == pytest.approx(2030.0)
        mag.check_validity(start)
        mag.check_validity(end)

class TestPoles:
    """Eq. (11) divides by cos(phi'), and eq. (16) multiplies by sec(phi')."""

    @pytest.mark.parametrize("lat", [90.0, -90.0])
    def test_exactly_on_a_pole_refuses(self, lat):
        with pytest.raises(mag.AtGeographicPole):
            mag.variation(lat, 0.0, decimal_year=2026.0)

    def test_near_the_pole_still_computes(self):
        """The guard is a singularity check, not a wide exclusion zone."""
        value = mag.variation(89.9, 0.0, decimal_year=2026.0)
        assert -180.0 < value < 180.0


class TestUsAirspaceSanity:
    """Variation across the US should match what sectionals show."""

    def test_west_coast_is_easterly(self):
        """California runs roughly 12-14 deg east."""
        value = mag.variation(37.36, -121.93, decimal_year=2026.0)  # near KSJC
        assert 10.0 < value < 16.0

    def test_east_coast_is_westerly(self):
        """The northeast runs meaningfully west."""
        value = mag.variation(40.78, -73.87, decimal_year=2026.0)  # near KLGA
        assert -18.0 < value < -8.0

    def test_agonic_line_crosses_the_country(self):
        """Somewhere in between, variation passes through zero."""
        west = mag.variation(37.0, -122.0, decimal_year=2026.0)
        east = mag.variation(40.0, -74.0, decimal_year=2026.0)
        assert west > 0.0 > east


class TestConversions:
    def test_decimal_year(self):
        assert mag.decimal_year_for(2026, 1, 1) == pytest.approx(2026.0)
        assert mag.decimal_year_for(2026, 7, 2) == pytest.approx(2026.5, abs=0.01)

    def test_east_is_least(self):
        """Easterly variation subtracts from a true course."""
        assert mag.true_to_magnetic(90.0, 14.0) == pytest.approx(76.0)

    def test_west_is_best(self):
        """Westerly variation adds."""
        assert mag.true_to_magnetic(90.0, -13.0) == pytest.approx(103.0)

    def test_conversions_are_inverse(self):
        for true_course in (0.0, 45.0, 180.0, 359.0):
            for var in (-20.0, 0.0, 15.0):
                magnetic = mag.true_to_magnetic(true_course, var)
                assert mag.magnetic_to_true(magnetic, var) == pytest.approx(true_course)

    def test_wraps_around_north(self):
        assert mag.true_to_magnetic(5.0, 14.0) == pytest.approx(351.0)
        assert mag.magnetic_to_true(355.0, -14.0) == pytest.approx(341.0)
