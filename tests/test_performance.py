"""POH performance tests.

The central test here is `TestGridPointsAreExact`: interpolating at a
published grid point must return the published number unchanged. That catches
reshaping mistakes -- a transposed axis or a mismatched temperature column
would still produce plausible numbers, but not the right ones.
"""

import bisect
import csv
import math
from itertools import pairwise
from pathlib import Path

import pytest

from engine import atmosphere as atm
from engine import performance as perf
from engine.atmosphere import isa_temperature_c

DATA = Path(__file__).resolve().parent.parent / "data" / "poh" / "c172s"


def _oat(pressure_altitude_ft: float, isa_dev_c: float = 0.0) -> float:
    """Absolute OAT for an ISA deviation at an altitude.

    The cruise chart's columns are ISA deviations, but `perf.cruise` takes an
    absolute temperature, so tests that mean "a standard day" have to say which
    altitude they mean it at.
    """
    return isa_temperature_c(pressure_altitude_ft) + isa_dev_c


def _raw(name: str) -> list[dict[str, str]]:
    with (DATA / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class TestGridPointsAreExact:
    """Every published cell must come back out of the interpolator unchanged."""

    def test_every_takeoff_cell(self):
        for row in _raw("takeoff"):
            weight, palt = float(row["wt"]), float(row["p_alt"])
            for temp in (0, 10, 20, 30, 40):
                result = perf.takeoff_distance(weight, palt, float(temp))
                assert result.ground_roll_ft == pytest.approx(
                    float(row[f"groundroll_{temp}"])
                ), f"ground roll at {weight} lb / {palt} ft / {temp} C"
                assert result.total_over_50ft_ft == pytest.approx(
                    float(row[f"clearfifty_{temp}"])
                ), f"50 ft distance at {weight} lb / {palt} ft / {temp} C"

    def test_every_landing_cell(self):
        for row in _raw("landing"):
            palt = float(row["p_alt"])
            for temp in (0, 10, 20, 30, 40):
                result = perf.landing_distance(palt, float(temp))
                assert result.ground_roll_ft == pytest.approx(
                    float(row[f"groundroll_{temp}"])
                )
                assert result.total_over_50ft_ft == pytest.approx(
                    float(row[f"clearfifty_{temp}"])
                )

    def test_every_climb_rate_cell(self):
        for row in _raw("max_climb_rate"):
            palt = float(row["p_alt"])
            for temp in (-20, 0, 20, 40):
                published = row[f"t_{temp}"]
                if not published.strip():
                    continue  # the blank 12000 ft / 40 C cell
                result = perf.climb_rate(palt, float(temp))
                assert result.fpm == pytest.approx(float(published))
                assert result.kias == pytest.approx(float(row["kias"]))

    def test_every_cruise_cell(self):
        for row in _raw("cruise"):
            alt, dev = float(row["press_alt"]), float(row["isa_dev_c"])
            result = perf.cruise(alt, float(row["rpm"]), _oat(alt, dev))
            label = f"{row['press_alt']} ft / {row['rpm']} RPM / ISA{dev:+.0f}"
            assert result.percent_power == pytest.approx(float(row["pwr"])), label
            assert result.ktas == pytest.approx(float(row["ktas"])), label
            assert result.gph == pytest.approx(float(row["gph"])), label


class TestRefusesToExtrapolate:
    """Outside the published envelope the engine must refuse, not guess."""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda: perf.takeoff_distance(2600, 0, 20), id="over-gross"),
            pytest.param(lambda: perf.takeoff_distance(2550, 9000, 20), id="too-high"),
            # A pressure altitude below sea level is *not* here: it is floored
            # onto the bottom row instead of refused. See TestBelowSeaLevel.
            pytest.param(lambda: perf.takeoff_distance(2550, 0, 45), id="too-hot"),
            pytest.param(lambda: perf.takeoff_distance(2550, 0, -5), id="too-cold"),
            pytest.param(lambda: perf.landing_distance(9000, 20), id="landing-high"),
            pytest.param(lambda: perf.climb_rate(13000, 0), id="climb-high"),
            pytest.param(lambda: perf.cruise(6000, 2400, 40), id="cruise-hot"),
            pytest.param(
                lambda: perf.cruise(14000, 2400, _oat(14000)), id="cruise-high"
            ),
            pytest.param(
                lambda: perf.cruise(6000, 2800, _oat(6000)), id="cruise-rpm-high"
            ),
        ],
    )
    def test_out_of_range_refuses(self, call):
        with pytest.raises(perf.OutsidePOHEnvelope):
            call()

    def test_blank_chart_cell_refuses(self):
        """The POH omits climb rate at 12000 ft / 40 C -- so do we."""
        with pytest.raises(perf.OutsidePOHEnvelope, match="blank"):
            perf.climb_rate(12000, 40)

    def test_cells_adjacent_to_the_blank_also_refuse(self):
        """Interpolating toward a hole would silently invent data."""
        with pytest.raises(perf.OutsidePOHEnvelope):
            perf.climb_rate(11500, 35)

    def test_hole_in_ragged_cruise_grid_refuses(self):
        """2700 RPM is published at 8000-10000 ft only, not down low."""
        perf.cruise(9000, 2700, _oat(9000))  # published, must work
        with pytest.raises(perf.OutsidePOHEnvelope):
            perf.cruise(2000, 2700, _oat(2000))


class TestCruiseSequence:
    """Cruise reads temperature, then RPM within a page, then between pages."""

    @staticmethod
    def _page(alt, rpm, isa_dev):
        """Read one altitude page by hand, in the prescribed order."""
        sub = [r for r in _raw("cruise") if float(r["press_alt"]) == alt]
        devs = sorted({float(r["isa_dev_c"]) for r in sub})
        rpms = sorted({float(r["rpm"]) for r in sub})
        cell = {
            (float(r["rpm"]), float(r["isa_dev_c"])): (
                float(r["pwr"]), float(r["ktas"]), float(r["gph"])
            )
            for r in sub
        }
        j = min(max(bisect.bisect_right(devs, isa_dev) - 1, 0), len(devs) - 2)
        tf = (isa_dev - devs[j]) / (devs[j + 1] - devs[j])
        by_rpm = {
            r: tuple(
                a + tf * (b - a)
                for a, b in zip(cell[(r, devs[j])], cell[(r, devs[j + 1])])
            )
            for r in rpms
        }
        i = min(max(bisect.bisect_right(rpms, rpm) - 1, 0), len(rpms) - 2)
        rf = (rpm - rpms[i]) / (rpms[i + 1] - rpms[i])
        return tuple(
            a + rf * (b - a) for a, b in zip(by_rpm[rpms[i]], by_rpm[rpms[i + 1]])
        )

    def test_matches_the_sequence_done_by_hand(self):
        """7000 ft between pages, 2450 RPM between settings, ISA+5 between columns."""
        lo, hi = self._page(6000, 2450, 5), self._page(8000, 2450, 5)
        want = [a + 0.5 * (b - a) for a, b in zip(lo, hi)]
        got = perf.cruise(7000, 2450, _oat(7000, 5))
        assert [got.percent_power, got.ktas, got.gph] == pytest.approx(want)

    @pytest.mark.parametrize(
        ("alt", "rpm", "culprit"),
        [
            (2000, 2700, 2000),  # above what the 2000 ft page publishes
            (6000, 2100, 6000),  # below what the 6000 ft page publishes
            (7000, 2700, 6000),  # between pages: the lower one refuses
            (5000, 2100, 6000),  # between pages: the upper one refuses
            (12000, 2700, 12000),  # the top page stops at 2650
        ],
    )
    def test_a_setting_missing_from_a_page_is_refused(self, alt, rpm, culprit):
        """Not a hole to interpolate around: the POH omits it on purpose."""
        with pytest.raises(perf.OutsidePOHEnvelope, match=f"at {culprit} ft"):
            perf.cruise(alt, rpm, _oat(alt))

    @pytest.mark.parametrize(
        ("alt", "rpm"), [(2000, 2100), (4000, 2100), (8000, 2700), (10000, 2700)]
    )
    def test_a_setting_at_the_edge_of_its_page_is_accepted(self, alt, rpm):
        """Sitting exactly on a page, its neighbour's narrower list cannot veto."""
        assert perf.cruise(alt, rpm, _oat(alt)).ktas > 0

    def test_available_rpm_is_the_intersection_between_pages(self):
        """Between pages only settings both publish can be read."""
        at_5000 = perf.available_cruise_rpm(5000, _oat(5000))
        assert 2100.0 not in at_5000  # 4000 has it, 6000 does not
        assert 2600.0 in at_5000  # both pages have it
        for rpm in at_5000:
            perf.cruise(5000, rpm, _oat(5000))  # must not raise

    def test_altitude_outside_the_chart_refuses(self):
        with pytest.raises(perf.OutsidePOHEnvelope, match="pressure altitude"):
            perf.cruise(14000, 2400, _oat(14000))

    def test_temperature_outside_the_chart_refuses(self):
        with pytest.raises(perf.OutsidePOHEnvelope, match="temperature"):
            perf.cruise(6000, 2400, 40)


class TestCruiseAtDensity:
    """Reading a too-hot or too-cold cruise point at an equal density.

    The chart publishes ISA-20, ISA and ISA+20 only, which a California
    afternoon leaves without trying. The substitution moves the query along a
    line of constant density onto a page that does publish it -- so every
    number still comes from inside the grid, and nothing is extrapolated.
    """

    def test_in_band_is_the_plain_chart_untouched(self):
        direct = perf.cruise(6000, 2400, _oat(6000, 10))
        lookup = perf.cruise_at_density(6000, 2400, _oat(6000, 10))
        assert lookup.substituted is False
        assert lookup.point == direct
        assert lookup.pressure_altitude_ft == 6000
        assert lookup.oat_c == pytest.approx(_oat(6000, 10))

    def test_hot_day_is_read_at_its_density_altitude_at_standard_temperature(self):
        """Step 2: at ISA the density altitude *is* the pressure altitude."""
        oat = _oat(6500, 25)
        lookup = perf.cruise_at_density(6500, 2400, oat)
        assert lookup.substituted is True
        assert lookup.density_altitude_ft == pytest.approx(
            atm.density_altitude(6500, oat)
        )
        # Read at that density altitude, on a standard day.
        assert lookup.pressure_altitude_ft == pytest.approx(lookup.density_altitude_ft)
        assert lookup.isa_deviation_c == pytest.approx(0.0, abs=1e-9)
        assert lookup.point == perf.cruise(
            lookup.density_altitude_ft, 2400, _oat(lookup.density_altitude_ft)
        )

    def test_density_off_the_top_falls_back_to_the_highest_covering_page(self):
        """Step 3: no standard-temperature page exists above 12000 ft."""
        ceiling = perf.cruise_altitude_range()[1]
        oat = atm.oat_for_density_altitude(11000, 13500)
        lookup = perf.cruise_at_density(11000, 2400, oat)
        assert lookup.substituted is True
        assert lookup.density_altitude_ft == pytest.approx(13500)
        assert lookup.pressure_altitude_ft == ceiling
        # The temperature is solved for, not guessed: reading it back at the
        # page it was solved on must return the density we started with.
        assert atm.density_altitude(ceiling, lookup.oat_c) == pytest.approx(13500)
        assert 0 < lookup.isa_deviation_c <= 20

    def test_cold_day_below_the_chart_floor_is_read_at_a_covering_page(self):
        """The same search running the other way, which step 3 gets for free."""
        oat = _oat(4000, -30)
        lookup = perf.cruise_at_density(4000, 2400, oat)
        assert lookup.substituted is True
        assert lookup.density_altitude_ft < perf.cruise_altitude_range()[0]
        assert atm.density_altitude(
            lookup.pressure_altitude_ft, lookup.oat_c
        ) == pytest.approx(lookup.density_altitude_ft)

    def test_air_no_page_covers_is_still_refused(self):
        """The point of the exercise is not to answer everything."""
        with pytest.raises(perf.OutsidePOHEnvelope, match="density altitude"):
            perf.cruise_at_density(12000, 2400, _oat(12000, 25))

    def test_an_rpm_the_substituted_page_omits_is_still_refused(self):
        """The chart is ragged, and moving pages does not make 2100 RPM exist.

        A setting the POH leaves off a page is left off deliberately, and the
        substitution has no standing to put it back.
        """
        with pytest.raises(perf.OutsidePOHEnvelope, match="RPM"):
            perf.cruise_at_density(2000, 2100, _oat(2000, 35))

    def test_hotter_air_keeps_costing_speed_across_the_seam(self):
        """True airspeed must not jump upward where the substitution starts.

        The step from the last published column to the first substituted
        reading is where a sign error would show, so it is checked across the
        boundary rather than only well inside it.
        """
        speeds = [
            perf.cruise_at_density(6000, 2400, _oat(6000, dev)).point.ktas
            for dev in (18, 20, 22, 25, 30)
        ]
        assert speeds == sorted(speeds, reverse=True)

    def test_fuel_flow_falls_with_temperature_once_substituted(self):
        flows = [
            perf.cruise_at_density(6000, 2400, _oat(6000, dev)).point.gph
            for dev in (22, 25, 30, 35)
        ]
        assert flows == sorted(flows, reverse=True)

    def test_the_step_at_the_seam_is_the_size_of_the_approximation(self):
        """What the equal-density reading costs, measured rather than assumed.

        The chart's own ISA+20 column and the standard-temperature page at the
        same density altitude do not agree exactly -- cruise performance is
        not a pure function of density -- so a plan jumps a little where the
        substitution takes over. It jumps by about 1%, in either direction,
        which is what makes this a defensible fallback and not a free lunch.
        A regression that widened it would show here.
        """
        for altitude in (2000, 4000, 6000, 8000):
            for rpm in (2300, 2400, 2500):
                published = perf.cruise(altitude, rpm, _oat(altitude, 20))
                # The same air, a whisker past the published column.
                substituted = perf.cruise_at_density(
                    altitude, rpm, _oat(altitude, 20.001)
                )
                assert substituted.substituted is True
                assert substituted.point.gph == pytest.approx(
                    published.gph, rel=0.01
                )
                assert substituted.point.ktas == pytest.approx(
                    published.ktas, rel=0.01
                )


class TestCruiseDensityBands:
    """The density altitudes each published cruise page covers."""

    def test_a_page_brackets_its_own_altitude(self):
        low, high = perf.cruise_density_range(6000)
        assert low < 6000 < high
        assert low == pytest.approx(atm.density_altitude(6000, _oat(6000, -20)))
        assert high == pytest.approx(atm.density_altitude(6000, _oat(6000, 20)))

    def test_one_band_per_published_page_lowest_first(self):
        pages = perf.cruise_density_pages()
        alts = [p[0] for p in pages]
        assert alts == sorted(alts)
        assert (alts[0], alts[-1]) == perf.cruise_altitude_range()

    def test_the_bands_overlap_so_their_union_has_no_gaps(self):
        """What makes the search in `cruise_at_density` always terminate.

        The pages are 2000 ft apart and each band spans about 4750 ft of
        density altitude, so consecutive bands overlap by more than half.
        """
        pages = perf.cruise_density_pages()
        for (_, _, high), (_, next_low, _) in pairwise(pages):
            assert next_low < high

    def test_every_band_edge_is_readable(self):
        """A query landing exactly on an edge must not fall between stools.

        The round trip through density and back into a temperature is not bit
        exact, so an edge lands a fraction of a picodegree outside the
        published column about half the time. Refusing those would put an
        unreachable seam in the middle of the covered range.
        """
        for altitude, low, high in perf.cruise_density_pages():
            for density in (low, high):
                oat = atm.oat_for_density_altitude(altitude, density)
                lookup = perf.cruise_at_density(altitude, 2400, oat)
                assert lookup.point.gph > 0

    def test_the_top_of_the_chart_is_reachable_from_another_page(self):
        """The clamp exercised where it actually bites: through step 3.

        The thinnest air the chart covers sits at the top page's ISA+20 corner,
        and a query arriving there from a lower pressure altitude has to solve
        for exactly that corner. Without the clamp the solved temperature
        overshoots it by rounding and the whole reading is refused.
        """
        ceiling = perf.cruise_altitude_range()[1]
        thinnest = perf.cruise_density_range(ceiling)[1]
        oat = atm.oat_for_density_altitude(ceiling - 1000, thinnest)
        lookup = perf.cruise_at_density(ceiling - 1000, 2400, oat)
        assert lookup.substituted is True
        assert lookup.pressure_altitude_ft == ceiling
        assert lookup.isa_deviation_c == pytest.approx(20.0)


class TestInterpolationScheme:
    """Altitude blends geometrically for distances; everything else linearly."""

    @staticmethod
    def _published(field, weight, palt, temp):
        for row in _raw("takeoff"):
            if float(row["wt"]) == weight and float(row["p_alt"]) == palt:
                return float(row[f"{field}_{temp}"])
        raise AssertionError("no such published cell")

    @pytest.mark.parametrize("palt", [500, 1500, 4500, 7500])
    @pytest.mark.parametrize("field", ["groundroll", "clearfifty"])
    def test_takeoff_altitude_is_log_linear(self, field, palt):
        """Halfway between two altitude rows must be the geometric mean."""
        lo = self._published(field, 2550.0, palt - 500, 20)
        hi = self._published(field, 2550.0, palt + 500, 20)
        result = perf.takeoff_distance(2550, palt, 20)
        got = result.ground_roll_ft if field == "groundroll" else result.total_over_50ft_ft
        assert got == pytest.approx(math.sqrt(lo * hi))
        # And that is genuinely different from a straight chord.
        assert got < (lo + hi) / 2

    def test_landing_altitude_is_log_linear(self):
        rows = {float(r["p_alt"]): r for r in _raw("landing")}
        lo, hi = float(rows[0]["groundroll_20"]), float(rows[1000]["groundroll_20"])
        assert perf.landing_distance(500, 20).ground_roll_ft == pytest.approx(
            math.sqrt(lo * hi)
        )

    def test_temperature_stays_linear(self):
        """The charts are straight in temperature, so the blend is arithmetic."""
        lo = self._published("groundroll", 2550.0, 4000, 10)
        hi = self._published("groundroll", 2550.0, 4000, 20)
        assert perf.takeoff_distance(2550, 4000, 15).ground_roll_ft == pytest.approx(
            (lo + hi) / 2
        )

    def test_climb_rate_altitude_stays_linear(self):
        """Climb rate falls towards the ceiling; a log blend would read low."""
        rows = {float(r["p_alt"]): r for r in _raw("max_climb_rate")}
        lo, hi = float(rows[0]["t_0"]), float(rows[2000]["t_0"])
        assert perf.climb_rate(1000, 0).fpm == pytest.approx((lo + hi) / 2)

    def test_log_interpolation_reads_below_the_chord_everywhere(self):
        """The altitude curve is convex, so geometric must undercut linear."""
        for palt in range(250, 8000, 500):
            got = perf.takeoff_distance(2550, palt, 20).ground_roll_ft
            floor_ = 1000 * (palt // 1000)
            lo = self._published("groundroll", 2550.0, floor_, 20)
            hi = self._published("groundroll", 2550.0, floor_ + 1000, 20)
            frac = (palt - floor_) / 1000
            assert got <= lo + frac * (hi - lo) + 1e-9


class TestPhysicalBehaviour:
    """Sanity properties that must hold regardless of the exact numbers."""

    def test_takeoff_distance_grows_with_altitude(self):
        low = perf.takeoff_distance(2550, 0, 20).ground_roll_ft
        high = perf.takeoff_distance(2550, 6000, 20).ground_roll_ft
        assert high > low

    def test_takeoff_distance_grows_with_temperature(self):
        cool = perf.takeoff_distance(2550, 0, 0).ground_roll_ft
        hot = perf.takeoff_distance(2550, 0, 40).ground_roll_ft
        assert hot > cool

    def test_takeoff_distance_grows_with_weight(self):
        light = perf.takeoff_distance(2200, 0, 20).ground_roll_ft
        heavy = perf.takeoff_distance(2550, 0, 20).ground_roll_ft
        assert heavy > light

    def test_takeoff_weight_rounds_up_to_published_chart(self):
        """An in-between weight reads the next heavier chart, not a blend."""
        for weight, chart in ((2201.0, 2400.0), (2400.0, 2400.0), (2401.0, 2550.0)):
            assert perf.takeoff_distance(weight, 2000, 20) == perf.takeoff_distance(
                chart, 2000, 20
            ), f"{weight} lb should read the {chart} lb chart"

    def test_takeoff_weight_below_lightest_chart_rounds_up(self):
        assert perf.takeoff_distance(2100, 0, 20) == perf.takeoff_distance(2200, 0, 20)

    def test_takeoff_weight_defaults_to_gross(self):
        assert perf.takeoff_distance(
            pressure_altitude_ft=2000, oat_c=20
        ) == perf.takeoff_distance(perf.MAX_GROSS_WEIGHT_LB, 2000, 20)

    def test_fifty_foot_distance_exceeds_ground_roll(self):
        result = perf.takeoff_distance(2550, 2000, 20)
        assert result.total_over_50ft_ft > result.ground_roll_ft

    def test_climb_rate_falls_with_altitude(self):
        assert perf.climb_rate(0, 0).fpm > perf.climb_rate(10000, 0).fpm

    def test_climb_rate_falls_with_temperature(self):
        assert perf.climb_rate(4000, -20).fpm > perf.climb_rate(4000, 20).fpm

    def test_fixed_rpm_loses_power_with_altitude(self):
        """A normally-aspirated engine cannot hold power as it climbs.

        At a fixed 2400 RPM, percent power, true airspeed and fuel flow all
        fall with altitude. This is the opposite of the "TAS rises with
        altitude" shorthand, which only holds at constant power.
        """
        low = perf.cruise(4000, 2400, _oat(4000))
        high = perf.cruise(10000, 2400, _oat(10000))
        assert high.percent_power < low.percent_power
        assert high.ktas < low.ktas
        assert high.gph < low.gph

    def test_matched_power_gives_more_speed_up_high(self):
        """The real reason to fly high: same power and fuel, more speed.

        Holding roughly 57% power costs about 8.2 gph at both altitudes, but
        buys noticeably more true airspeed in the thinner air.
        """
        low = perf.cruise(2000, 2300, _oat(2000))
        high = perf.cruise(10000, 2500, _oat(10000))
        assert low.percent_power == pytest.approx(high.percent_power, abs=1.0)
        assert high.ktas > low.ktas
        assert high.gph == pytest.approx(low.gph, abs=0.3)

    def test_cruise_fuel_flow_grows_with_rpm(self):
        assert (
            perf.cruise(6000, 2500, _oat(6000)).gph
            > perf.cruise(6000, 2300, _oat(6000)).gph
        )


class TestWindAndSurfaceCorrections:
    def test_headwind_shortens_takeoff(self):
        still = perf.takeoff_distance(2550, 0, 20).ground_roll_ft
        headwind = perf.takeoff_distance(2550, 0, 20, headwind_kt=9).ground_roll_ft
        # The POH note is 10% per 9 knots.
        assert headwind == pytest.approx(still * 0.9)

    def test_tailwind_lengthens_takeoff(self):
        still = perf.takeoff_distance(2550, 0, 20).ground_roll_ft
        tailwind = perf.takeoff_distance(2550, 0, 20, headwind_kt=-2).ground_roll_ft
        # The POH note is 10% per 2 knots of tailwind.
        assert tailwind == pytest.approx(still * 1.1)

    def test_grass_penalises_takeoff_by_15_percent_of_roll(self):
        paved = perf.takeoff_distance(2550, 0, 20)
        grass = perf.takeoff_distance(2550, 0, 20, dry_grass=True)
        assert grass.ground_roll_ft == pytest.approx(paved.ground_roll_ft * 1.15)

    def test_grass_penalises_landing_by_45_percent_of_roll(self):
        paved = perf.landing_distance(0, 20)
        grass = perf.landing_distance(0, 20, dry_grass=True)
        assert grass.ground_roll_ft == pytest.approx(paved.ground_roll_ft * 1.45)


class TestClimbSegments:
    def test_segment_is_difference_of_cumulative_rows(self):
        """Climbing 0->8000 must equal 0->4000 plus 4000->8000."""
        whole = perf.climb_from_to(0, 8000)
        first = perf.climb_from_to(0, 4000)
        second = perf.climb_from_to(4000, 8000)
        assert whole.time_min == pytest.approx(first.time_min + second.time_min)
        assert whole.fuel_gal == pytest.approx(first.fuel_gal + second.fuel_gal)

    def test_zero_climb_costs_nothing(self):
        segment = perf.climb_from_to(3000, 3000)
        assert segment.time_min == pytest.approx(0.0)
        assert segment.fuel_gal == pytest.approx(0.0)

    def test_descending_segment_rejected(self):
        with pytest.raises(ValueError):
            perf.climb_from_to(6000, 2000)

    def test_hot_day_costs_more(self):
        standard = perf.climb_from_to(0, 6000)
        hot = perf.climb_from_to(0, 6000, oat_c=isa_temperature_c(3000) + 20)
        # The POH note is 10% per 10 degC above standard, so 20 degC is +20%.
        assert hot.time_min == pytest.approx(standard.time_min * 1.2)
        assert hot.fuel_gal == pytest.approx(standard.fuel_gal * 1.2)

    def test_cold_day_costs_less(self):
        """The note runs both ways: 10% off per 10 degC below standard."""
        standard = perf.climb_from_to(0, 6000)
        cold = perf.climb_from_to(0, 6000, oat_c=isa_temperature_c(3000) - 20)
        assert cold.time_min == pytest.approx(standard.time_min * 0.8)
        assert cold.fuel_gal == pytest.approx(standard.fuel_gal * 0.8)

    def test_an_absurdly_cold_day_still_takes_time_to_climb(self):
        """The correction is floored: no forecast drives a climb to zero."""
        standard = perf.climb_from_to(0, 6000)
        frigid = perf.climb_from_to(0, 6000, oat_c=isa_temperature_c(3000) - 90)
        assert frigid.time_min == pytest.approx(standard.time_min * 0.5)

    def test_climb_speed_is_the_average_of_the_rows_it_spans(self):
        """The table prints 74 KIAS at sea level and 72 at 8000 ft."""
        rows = {float(r["p_alt"]): float(r["speed"]) for r in _raw("climb_dist")}
        segment = perf.climb_from_to(0, 8000)
        assert segment.kias == pytest.approx((rows[0.0] + rows[8000.0]) / 2)

    def test_climb_speed_at_a_single_altitude_is_that_row(self):
        rows = {float(r["p_alt"]): float(r["speed"]) for r in _raw("climb_dist")}
        assert perf.climb_from_to(4000, 4000).kias == pytest.approx(rows[4000.0])


class TestAvailableCruiseRpm:
    def test_low_altitude_offers_low_rpm(self):
        assert 2100.0 in perf.available_cruise_rpm(2000, _oat(2000))

    def test_low_altitude_excludes_high_rpm(self):
        assert 2700.0 not in perf.available_cruise_rpm(2000, _oat(2000))

    def test_high_altitude_excludes_low_rpm(self):
        assert 2100.0 not in perf.available_cruise_rpm(12000, _oat(12000, -20))

    def test_all_returned_settings_actually_work(self):
        for rpm in perf.available_cruise_rpm(8000, _oat(8000)):
            result = perf.cruise(8000, rpm, _oat(8000))
            assert not math.isnan(result.ktas)
            assert result.gph > 0


class TestBelowSeaLevel:
    """A high altimeter setting puts a low field under the charts' bottom row.

    Every chart with a pressure altitude axis starts at 0 ft, but pressure
    altitude does not. At a 4 ft field anything above about 29.93 inHg is
    already negative, so refusing would fail an ordinary high-pressure day.
    The bottom row is read instead -- conservative in every case, because air
    that dense makes the aeroplane beat what the chart says.
    """

    @pytest.mark.parametrize("below", [-1.0, -167.0, -500.0, -2000.0])
    def test_takeoff_reads_the_sea_level_row(self, below):
        low = perf.takeoff_distance(2550, below, 20)
        at_sea_level = perf.takeoff_distance(2550, 0.0, 20)
        assert low.ground_roll_ft == at_sea_level.ground_roll_ft
        assert low.total_over_50ft_ft == at_sea_level.total_over_50ft_ft

    @pytest.mark.parametrize("below", [-1.0, -167.0, -500.0, -2000.0])
    def test_landing_reads_the_sea_level_row(self, below):
        low = perf.landing_distance(below, 20)
        at_sea_level = perf.landing_distance(0.0, 20)
        assert low.ground_roll_ft == at_sea_level.ground_roll_ft
        assert low.total_over_50ft_ft == at_sea_level.total_over_50ft_ft

    @pytest.mark.parametrize("below", [-1.0, -167.0, -500.0, -2000.0])
    def test_the_sea_level_reading_is_reported_as_off_chart(self, below):
        """Conservative, but still not the row the query asked for."""
        for distance in (
            perf.takeoff_distance(2550, below, 20),
            perf.landing_distance(below, 20),
        ):
            assert distance.extrapolated
            assert not distance.optimistic
            assert distance.off_chart[0].what == "pressure altitude"

    def test_a_reading_on_the_chart_carries_nothing(self):
        distance = perf.takeoff_distance(2550, 2000.0, 20)
        assert distance.off_chart == ()
        assert not distance.extrapolated

    @pytest.mark.parametrize("below", [-1.0, -167.0, -500.0, -2000.0])
    def test_climb_rate_reads_the_sea_level_row(self, below):
        assert perf.climb_rate(below, 20) == perf.climb_rate(0.0, 20)

    def test_a_climb_starting_below_sea_level_starts_at_the_bottom_row(self):
        assert perf.climb_from_to(-167.0, 6000.0) == perf.climb_from_to(0.0, 6000.0)

    def test_a_climb_entirely_below_sea_level_costs_nothing(self):
        """Both ends land on the same row, so the segment is empty.

        Not an error: the aeroplane really is climbing, the POH just publishes
        nothing to charge it with, and zero is the only honest reading of a
        chart that starts where this climb ends.
        """
        segment = perf.climb_from_to(-400.0, -100.0)
        assert segment.time_min == pytest.approx(0.0)
        assert segment.fuel_gal == pytest.approx(0.0)

    def test_a_descending_climb_is_still_a_caller_error(self):
        """The floor must not turn a backwards segment into a valid one."""
        with pytest.raises(ValueError, match="end above where it starts"):
            perf.climb_from_to(-100.0, -400.0)

    def test_the_floor_is_conservative_for_takeoff(self):
        """Denser air shortens the roll, so the sea level row over-reads it."""
        denser = perf.takeoff_distance(2550, -500.0, 20)
        thinner = perf.takeoff_distance(2550, 500.0, 20)
        assert denser.ground_roll_ft < thinner.ground_roll_ft

    def test_temperature_is_not_floored(self):
        """Only the pressure altitude axis is clamped; the rest still refuse."""
        with pytest.raises(perf.OutsidePOHEnvelope):
            perf.climb_rate(-500.0, 60.0)
