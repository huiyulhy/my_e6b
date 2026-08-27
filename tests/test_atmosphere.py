"""Atmosphere tests, checked against published US Standard Atmosphere values."""

import math

import pytest

from engine import atmosphere as atm


class TestIsaConstants:
    """Pin the defining constants themselves, not just their consequences.

    POH performance charts and altimeters are calibrated to ISA, so these are
    the values that keep the engine in step with the book its numbers came
    from. A nearby-but-different model -- NASA Glenn's simplified curve fit
    uses 15.04 degC and 0.00649 K/m -- introduces a small systematic bias that
    compounds through pressure altitude into density altitude into every
    performance lookup. Asserting the constants makes that impossible to
    change by accident.
    """

    def test_sea_level_temperature_is_exactly_15c(self):
        assert atm.T0_C == 15.0
        assert atm.T0_K == pytest.approx(288.15)

    def test_lapse_rate_is_exactly_6_point_5_per_km(self):
        assert atm.LAPSE_K_PER_M == -0.0065

    def test_derived_constants_follow_from_the_defining_ones(self):
        assert atm.RHO0 == pytest.approx(1.2250, abs=1e-4)
        assert atm.T_TROPOPAUSE_K == pytest.approx(216.65, abs=0.01)
        # Pressure ratio delta = (T/T0) ** _DELTA_EXP, density ratio sigma
        # carries one fewer power of the temperature ratio.
        assert atm._DELTA_EXP == pytest.approx(5.2559, abs=1e-4)
        assert atm._SIGMA_EXP == pytest.approx(atm._DELTA_EXP - 1.0)


class TestStandardAtmosphere:
    """Reference values from the US Standard Atmosphere 1976 tables."""

    def test_sea_level_conditions(self):
        assert atm.isa_temperature_c(0) == pytest.approx(15.0)
        assert atm.isa_pressure_pa(0) == pytest.approx(101325.0)
        assert atm.isa_density(0) == pytest.approx(1.2250, abs=1e-4)

    def test_tropopause_conditions(self):
        tropopause_ft = atm.m_to_ft(11000.0)
        assert atm.isa_temperature_c(tropopause_ft) == pytest.approx(-56.5, abs=0.01)
        assert atm.isa_pressure_pa(tropopause_ft) == pytest.approx(22632.0, rel=1e-4)
        assert atm.isa_density(tropopause_ft) == pytest.approx(0.3639, abs=1e-4)

    def test_lapse_rate_matches_rule_of_thumb(self):
        """About 2 degC per 1000 ft, the number pilots actually use."""
        assert atm.isa_temperature_c(1000) == pytest.approx(15.0 - 1.9812, abs=1e-3)

    @pytest.mark.parametrize("altitude_ft", [0, 2500, 5000, 10000, 18000, 30000])
    def test_pressure_is_monotonically_decreasing(self, altitude_ft):
        assert atm.isa_pressure_pa(altitude_ft) > atm.isa_pressure_pa(altitude_ft + 500)

    @pytest.mark.parametrize("altitude_ft", [0, 3000, 8500, 20000, 36000])
    def test_pressure_inversion_round_trips(self, altitude_ft):
        pressure = atm.isa_pressure_pa(altitude_ft)
        assert atm.altitude_for_pressure_pa(pressure) == pytest.approx(altitude_ft, abs=0.1)


class TestModelCeiling:
    """The model covers the troposphere only, and says so rather than guessing.

    A 172S service ceiling is around 14000 ft, so the tropopause at 36089 ft
    is far beyond anything this project needs. Refusing above it keeps the
    module honest about what it has been checked against.
    """

    def test_ceiling_is_the_tropopause(self):
        assert atm.TROPOPAUSE_CEILING_FT == pytest.approx(36089, abs=1)

    def test_at_the_ceiling_still_works(self):
        assert atm.isa_temperature_c(36089) == pytest.approx(-56.5, abs=0.01)

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda: atm.isa_temperature_c(40000), id="temperature"),
            pytest.param(lambda: atm.isa_pressure_pa(40000), id="pressure"),
            pytest.param(lambda: atm.isa_density(40000), id="density"),
            pytest.param(lambda: atm.altitude_for_pressure_pa(10000.0), id="inverse-pressure"),
            pytest.param(lambda: atm.density_altitude(40000, -50), id="density-altitude"),
        ],
    )
    def test_above_the_ceiling_refuses(self, call):
        with pytest.raises(atm.AboveModelCeiling):
            call()

    def test_the_refusal_says_where_the_limit_is(self):
        with pytest.raises(atm.AboveModelCeiling, match="36089"):
            atm.isa_temperature_c(40000)


class TestPressureAltitude:
    def test_standard_setting_gives_field_elevation(self):
        """At 29.92, pressure altitude and field elevation coincide."""
        for elevation in (0, 1200, 5400):
            assert atm.pressure_altitude(elevation, atm.P0_INHG) == pytest.approx(
                elevation, abs=0.5
            )

    def test_low_altimeter_setting_raises_pressure_altitude(self):
        assert atm.pressure_altitude(1000, 29.42) > 1000

    def test_agrees_with_cockpit_rule_of_thumb(self):
        """The 1000 ft per inHg rule should be close, but not exact."""
        exact = atm.pressure_altitude(1000, 29.42)
        approx = atm.pressure_altitude_approx(1000, 29.42)
        assert exact == pytest.approx(approx, abs=60)
        assert exact != approx

    @pytest.mark.parametrize(
        ("elevation_ft", "altimeter_inhg"),
        [(0, 29.92126), (0, 28.92), (0, 30.92), (1000, 29.42), (5000, 30.42), (8000, 29.10)],
    )
    def test_inverts_the_altimeter_setting_definition(self, elevation_ft, altimeter_inhg):
        """Check against the definition of altimeter setting, not against algebra.

        A station derives its altimeter setting A from the measured station
        pressure and its surveyed elevation by the ICAO/NWS relation

            A^(1/n) = P_station^(1/n) + k * elev

        which is additive in p^(1/n), the quantity linear in altitude under
        the ISA power law. `pressure_altitude` must be the exact inverse: go
        forward from its answer to the station pressure, apply the definition,
        and the original setting has to come back.

        Deriving the expected PA algebraically instead would only re-check the
        arithmetic -- an earlier version of this test did exactly that, and
        confirmed a multiplicative model that was wrong by tens of feet.
        """
        n_inv = 1.0 / atm._DELTA_EXP
        scale_height_ft = atm.m_to_ft(-atm.T0_K / atm.LAPSE_K_PER_M)
        k = atm.P0_PA**n_inv / scale_height_ft

        pa = atm.pressure_altitude(elevation_ft, altimeter_inhg)
        station_pa = atm.isa_pressure_pa(pa)
        setting = (station_pa**n_inv + k * elevation_ft) ** atm._DELTA_EXP

        assert atm.pa_to_inhg(setting) == pytest.approx(altimeter_inhg, abs=1e-6)


class TestDensityAltitude:
    def test_standard_temperature_gives_pressure_altitude(self):
        """On a standard day, density altitude equals pressure altitude."""
        for pa in (0, 2000, 8000):
            oat = atm.isa_temperature_c(pa)
            assert atm.density_altitude(pa, oat) == pytest.approx(pa, abs=0.5)

    def test_hot_day_raises_density_altitude(self):
        assert atm.density_altitude(5000, 30) > 5000

    def test_cold_day_lowers_density_altitude(self):
        assert atm.density_altitude(5000, -10) < 5000

    def test_agrees_with_120ft_per_degree_rule(self):
        exact = atm.density_altitude(5000, 30)
        approx = atm.density_altitude_approx(5000, 30)
        assert exact == pytest.approx(approx, rel=0.03)

    def test_isa_deviation(self):
        assert atm.isa_deviation_c(0, 15) == pytest.approx(0.0)
        assert atm.isa_deviation_c(0, 30) == pytest.approx(15.0)


class TestOatForDensityAltitude:
    """The inverse of `density_altitude` in its temperature argument."""

    @pytest.mark.parametrize("pressure_altitude_ft", [0, 2000, 6500, 12000])
    @pytest.mark.parametrize("density_altitude_ft", [-500, 0, 4000, 14000])
    def test_round_trips_exactly(self, pressure_altitude_ft, density_altitude_ft):
        oat = atm.oat_for_density_altitude(pressure_altitude_ft, density_altitude_ft)
        assert atm.density_altitude(pressure_altitude_ft, oat) == pytest.approx(
            density_altitude_ft
        )

    def test_standard_temperature_where_the_two_altitudes_agree(self):
        for altitude in (0, 5000, 10000):
            assert atm.oat_for_density_altitude(altitude, altitude) == pytest.approx(
                atm.isa_temperature_c(altitude)
            )

    def test_thinner_air_at_a_fixed_pressure_altitude_means_hotter(self):
        colder = atm.oat_for_density_altitude(6000, 6000)
        hotter = atm.oat_for_density_altitude(6000, 9000)
        assert hotter > colder


class TestTemperatureProfile:
    """The reconciliation between a field METAR and a forecast aloft."""

    def test_no_samples_is_the_route_wide_deviation(self):
        profile = atm.TemperatureProfile.from_observations([], default_deviation_c=10.0)
        for pa in (0, 3000, 9000):
            assert profile.deviation_at(pa) == pytest.approx(10.0)
            assert profile.oat_at_pressure_altitude(pa) == pytest.approx(
                atm.isa_temperature_c(pa) + 10.0
            )

    def test_one_sample_wins_over_the_default_everywhere(self):
        profile = atm.TemperatureProfile.from_observations(
            [atm.TemperatureSample.observed(5000, 20.0)], default_deviation_c=-30.0
        )
        assert profile.deviation_at(5000) == pytest.approx(20.0 - atm.isa_temperature_c(5000))

    def test_extrapolation_is_flat_in_deviation_not_in_temperature(self):
        """The load-bearing property of the whole design.

        A single 5000 ft sample must still cool at the standard rate above and
        below it. Someone "simplifying" this into temperature space would hold
        20 degC all the way to 9000 ft, which is 8 degC of error and about a
        thousand feet of density altitude.
        """
        profile = atm.TemperatureProfile.from_observations(
            [atm.TemperatureSample.observed(5000, 20.0)]
        )
        dev = 20.0 - atm.isa_temperature_c(5000)
        for pa in (0, 5000, 9000):
            assert profile.oat_at_pressure_altitude(pa) == pytest.approx(
                atm.isa_temperature_c(pa) + dev
            )
        assert profile.oat_at_pressure_altitude(9000) < 20.0

    def test_interpolates_linearly_in_pressure_altitude(self):
        low = atm.TemperatureSample(0.0, 10.0)
        high = atm.TemperatureSample(10000.0, 0.0)
        profile = atm.TemperatureProfile.from_observations([low, high])
        assert profile.deviation_at(2500) == pytest.approx(7.5)
        assert profile.deviation_at(5000) == pytest.approx(5.0)

    def test_clamps_outside_the_observed_range(self):
        profile = atm.TemperatureProfile.from_observations(
            [atm.TemperatureSample(2000.0, 8.0), atm.TemperatureSample(8000.0, 2.0)]
        )
        assert profile.deviation_at(-500) == pytest.approx(8.0)
        assert profile.deviation_at(20000) == pytest.approx(2.0)

    def test_unsorted_input_is_ordered(self):
        profile = atm.TemperatureProfile.from_observations(
            [atm.TemperatureSample(8000.0, 2.0), atm.TemperatureSample(2000.0, 8.0)]
        )
        assert [s.pressure_altitude_ft for s in profile.samples] == [2000.0, 8000.0]

    def test_near_coincident_samples_are_averaged(self):
        """Two fields a couple of hundred feet apart are one slice of air.

        Left unmerged they would imply a gradient of tens of degrees per
        thousand feet across that sliver, and the flat-below extrapolation
        would carry the wrong end of it into the takeoff.
        """
        profile = atm.TemperatureProfile.from_observations(
            [atm.TemperatureSample(100.0, 18.0), atm.TemperatureSample(300.0, 8.0)]
        )
        assert len(profile.samples) == 1
        assert profile.samples[0].isa_deviation_c == pytest.approx(13.0)
        assert profile.samples[0].pressure_altitude_ft == pytest.approx(200.0)

    def test_samples_further_apart_than_the_tolerance_stay_separate(self):
        profile = atm.TemperatureProfile.from_observations(
            [atm.TemperatureSample(100.0, 18.0), atm.TemperatureSample(3000.0, 8.0)]
        )
        assert len(profile.samples) == 2

    def test_observed_uses_the_stations_own_altimeter(self):
        """A METAR's temperature belongs at the PA that station's setting gives."""
        sample = atm.TemperatureSample.observed(500, 20.0, altimeter_inhg=29.42)
        assert sample.pressure_altitude_ft == pytest.approx(
            atm.pressure_altitude(500, 29.42)
        )
        assert sample.oat_c == pytest.approx(20.0)

    def test_superadiabatic_pair_warns(self):
        profile = atm.TemperatureProfile.from_observations(
            [
                atm.TemperatureSample.observed(0, 35.0),
                atm.TemperatureSample.observed(3000, 20.0),
            ]
        )
        assert profile.lapse_warnings()

    def test_inversion_is_legal_and_silent(self):
        """A marine layer is ordinary weather, not a data-entry error."""
        profile = atm.TemperatureProfile.from_observations(
            [
                atm.TemperatureSample.observed(0, 12.0),
                atm.TemperatureSample.observed(3000, 18.0),
            ]
        )
        assert profile.lapse_warnings() == ()

    def test_standard_lapse_between_two_isa_samples_is_silent(self):
        profile = atm.TemperatureProfile.from_observations(
            [
                atm.TemperatureSample.observed(0, atm.isa_temperature_c(0)),
                atm.TemperatureSample.observed(9000, atm.isa_temperature_c(9000)),
            ]
        )
        assert profile.lapse_warnings() == ()
        assert profile.deviation_at(4500) == pytest.approx(0.0, abs=1e-9)


class TestRelativeRatios:
    def test_all_unity_at_sea_level_standard(self):
        assert atm.relative_pressure(0) == pytest.approx(1.0)
        assert atm.relative_temperature(atm.T0_C) == pytest.approx(1.0)
        assert atm.relative_density(0) == pytest.approx(1.0)

    @pytest.mark.parametrize("pa", [0, 2000, 8000])
    @pytest.mark.parametrize("oat", [-20, 0, 15, 35])
    def test_sigma_is_delta_over_theta(self, pa, oat):
        """The ideal gas identity, which holds off the standard profile too.

        Density altitude is the coordinate in which sigma is recoverable, so
        going conditions -> DA -> sigma must reproduce delta/Theta exactly.
        """
        expected = atm.relative_pressure(pa) / atm.relative_temperature(oat)
        assert atm.relative_density(atm.density_altitude(pa, oat)) == pytest.approx(
            expected, rel=1e-4
        )

    def test_theta_reads_a_temperature_not_an_altitude(self):
        """On a standard day Theta at altitude follows from the ISA temperature."""
        for h in (0, 5000, 10000):
            assert atm.relative_temperature(atm.isa_temperature_c(h)) == pytest.approx(
                atm.c_to_k(atm.isa_temperature_c(h)) / atm.T0_K
            )

    @pytest.mark.parametrize("da", [-1000, 0, 5000, 20000, 36000])
    def test_density_ratio_round_trips_through_its_inverse(self, da):
        assert atm.altitude_for_relative_density(
            atm.relative_density(da)
        ) == pytest.approx(da, abs=1e-6)

    def test_ratios_agree_with_the_si_functions(self):
        """The ratio forms must not drift from `isa_pressure_pa` / `isa_density`."""
        for h in (0, 3000, 12000):
            assert atm.relative_pressure(h) == pytest.approx(
                atm.isa_pressure_pa(h) / atm.P0_PA, rel=1e-4
            )
            assert atm.relative_density(h) == pytest.approx(
                atm.isa_density(h) / atm.RHO0, rel=1e-4
            )


class TestAirspeed:
    def test_tas_equals_cas_at_sea_level_standard(self):
        assert atm.tas_from_cas(100, 0) == pytest.approx(100.0, abs=0.01)

    def test_tas_exceeds_cas_with_altitude(self):
        tas = atm.tas_from_cas(100, atm.density_altitude(8000, atm.isa_temperature_c(8000)))
        assert tas > 100
        # The familiar rough figure is 2% per 1000 ft, so about 116 kt.
        assert tas == pytest.approx(113, abs=3)

    def test_colder_than_isa_gives_less_tas(self):
        """Denser air at the same pressure altitude means TAS closer to CAS."""
        cold = atm.tas_from_cas(110, atm.density_altitude(8000, -20))
        hot = atm.tas_from_cas(110, atm.density_altitude(8000, 20))
        assert cold < hot

    def test_zero_airspeed(self):
        assert atm.tas_from_cas(0, 5000) == 0.0


class TestUnitConversions:
    def test_round_trips(self):
        assert atm.m_to_ft(atm.ft_to_m(1234.5)) == pytest.approx(1234.5)
        assert atm.k_to_c(atm.c_to_k(15.0)) == pytest.approx(15.0)
        assert atm.pa_to_inhg(atm.inhg_to_pa(29.92)) == pytest.approx(29.92)

    def test_known_values(self):
        assert atm.ft_to_m(1000) == pytest.approx(304.8, abs=0.01)
        assert atm.inhg_to_pa(atm.P0_INHG) == pytest.approx(101325.0)
        assert not math.isnan(atm.isa_density(0))
