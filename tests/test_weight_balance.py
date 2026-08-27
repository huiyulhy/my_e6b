"""Weight and balance tests.

The arithmetic is a weighted mean, so the tests are mostly about the ways a
weighted mean can be asked for and cannot be answered.
"""

import pytest

from engine import weight_balance as wb


class TestStation:
    def test_moment_is_weight_times_arm(self):
        assert wb.Station("Fuel", 180.0, 48.0).moment_in_lb == pytest.approx(8640.0)

    def test_a_negative_weight_gives_a_negative_moment(self):
        # Fuel burned off is entered as a negative weight at the tank arm.
        assert wb.Station("Burn", -120.0, 48.0).moment_in_lb == pytest.approx(-5760.0)


class TestCompute:
    def test_a_single_station_sits_at_its_own_arm(self):
        loading = wb.compute([wb.Station("Empty", 1680.0, 39.0)])
        assert loading.gross_weight_lb == pytest.approx(1680.0)
        assert loading.cg_in == pytest.approx(39.0)

    def test_gross_weight_and_cg_of_a_typical_172s_loading(self):
        loading = wb.compute(
            [
                wb.Station("Empty weight", 1680.0, 39.1),
                wb.Station("Front seats", 340.0, 37.0),
                wb.Station("Rear seats", 170.0, 73.0),
                wb.Station("Fuel", 318.0, 48.0),
                wb.Station("Baggage", 40.0, 95.0),
            ]
        )
        assert loading.gross_weight_lb == pytest.approx(2548.0)
        assert loading.total_moment_in_lb == pytest.approx(
            1680.0 * 39.1 + 340.0 * 37.0 + 170.0 * 73.0 + 318.0 * 48.0 + 40.0 * 95.0
        )
        assert loading.cg_in == pytest.approx(
            loading.total_moment_in_lb / 2548.0
        )

    def test_cg_moves_aft_when_weight_is_added_aft_of_it(self):
        forward = wb.compute([wb.Station("Empty", 1680.0, 39.1)])
        with_baggage = wb.compute(
            [wb.Station("Empty", 1680.0, 39.1), wb.Station("Baggage", 100.0, 95.0)]
        )
        assert with_baggage.cg_in > forward.cg_in

    def test_a_zero_weight_station_changes_nothing(self):
        # An empty seat is a row on the form, not a mass: it must not drag the
        # CG toward its arm.
        alone = wb.compute([wb.Station("Empty", 1680.0, 39.1)])
        with_empty_seat = wb.compute(
            [wb.Station("Empty", 1680.0, 39.1), wb.Station("Rear seats", 0.0, 73.0)]
        )
        assert with_empty_seat.cg_in == pytest.approx(alone.cg_in)

    def test_negative_weights_take_fuel_back_out(self):
        full = wb.compute(
            [wb.Station("Empty", 1680.0, 39.1), wb.Station("Fuel", 318.0, 48.0)]
        )
        zero_fuel = wb.compute(
            [
                wb.Station("Empty", 1680.0, 39.1),
                wb.Station("Fuel", 318.0, 48.0),
                wb.Station("Burn", -318.0, 48.0),
            ]
        )
        assert zero_fuel.gross_weight_lb == pytest.approx(1680.0)
        assert zero_fuel.cg_in == pytest.approx(39.1)
        assert full.cg_in > zero_fuel.cg_in

    def test_stations_are_kept_in_the_result(self):
        loading = wb.compute([wb.Station("Empty", 1680.0, 39.1)])
        assert [station.name for station in loading.stations] == ["Empty"]

    def test_no_stations_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            wb.compute([])

    def test_zero_total_weight_is_refused(self):
        # A CG is a weighted mean; with no weight there is nothing to average.
        with pytest.raises(ValueError, match="greater than zero"):
            wb.compute([wb.Station("Empty seat", 0.0, 37.0)])

    def test_negative_total_weight_is_refused(self):
        with pytest.raises(ValueError, match="greater than zero"):
            wb.compute(
                [wb.Station("Fuel", 100.0, 48.0), wb.Station("Burn", -200.0, 48.0)]
            )
