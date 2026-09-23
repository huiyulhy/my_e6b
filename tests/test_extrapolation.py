"""Nothing is extrapolated, and everything read off the chart says so.

The rule this file guards is one sentence: a query past the edge of a POH
chart is either refused or answered from a published cell beside it, and the
second case is never silent. What varies is how far the consequence travels --
from the chart layer, through the runway check, onto the navlog row, and into
the single word at the top of the go/no-go.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine import navlog as nl
from engine import performance as perf
from engine import preflight as pf
from engine.geo import LatLon

SEA_LEVEL = {
    "elevation_ft": 0.0,
    "oat_c": 15.0,
    "pressure_altitude_ft": 0.0,
    "density_altitude_ft": 0.0,
    "weight_lb": 2550.0,
    "margin": 0.2,
}


def check(runways, wind=None, operation="takeoff", **overrides):
    return pf.check_airport(
        airport="KTST",
        operation=operation,
        runways=tuple(runways),
        wind=wind,
        **{**SEA_LEVEL, **overrides},
    )


def fuel(passes=True):
    return pf.check_fuel(
        fuel_on_board_gal=50.0 if passes else 4.0,
        burn_gal=10.0,
        reserve_required_gal=4.0,
        reserve_minutes=30.0,
        margin=0.1,
        night=False,
    )


# --- the rule itself -----------------------------------------------------


class TestNothingIsExtrapolated:
    """Past the edge, the answer is a refusal or a published cell. Never a fit."""

    def test_above_the_heaviest_chart_is_refused(self):
        with pytest.raises(perf.OutsidePOHEnvelope):
            perf.takeoff_distance(2600.0, 0.0, 20.0)

    def test_above_the_top_of_the_altitude_axis_is_refused(self):
        with pytest.raises(perf.OutsidePOHEnvelope):
            perf.takeoff_distance(2550.0, 20000.0, 20.0)

    def test_off_the_end_of_the_temperature_axis_is_refused(self):
        with pytest.raises(perf.OutsidePOHEnvelope):
            perf.landing_distance(0.0, 60.0)

    def test_a_tailwind_past_the_correction_is_refused(self):
        with pytest.raises(perf.OutsidePOHEnvelope, match="tailwind"):
            perf.takeoff_distance(2550.0, 0.0, 20.0, headwind_kt=-25.0)

    def test_a_density_altitude_no_page_covers_is_refused(self):
        with pytest.raises(perf.OutsidePOHEnvelope):
            perf.cruise_at_density(12000.0, 2400.0, 60.0)


# --- the chart layer reports what it substituted -------------------------


class TestTheChartLayerSaysSo:
    def test_a_reading_on_the_chart_claims_nothing(self):
        distance = perf.takeoff_distance(2550.0, 2000.0, 20.0)
        assert distance.off_chart == ()
        assert not distance.extrapolated
        assert not distance.optimistic

    def test_below_the_bottom_row_is_reported_and_conservative(self):
        distance = perf.takeoff_distance(2550.0, -300.0, 20.0)
        assert distance.extrapolated
        assert not distance.optimistic
        assert distance.off_chart[0].what == "pressure altitude"
        assert "errs long" in distance.off_chart[0].detail

    def test_a_headwind_past_the_correction_is_capped_and_reported(self):
        distance = perf.landing_distance(0.0, 20.0, headwind_kt=45.0)
        capped = perf.landing_distance(0.0, 20.0, headwind_kt=30.0)
        assert distance.ground_roll_ft == pytest.approx(capped.ground_roll_ft)
        assert [e.what for e in distance.off_chart] == ["headwind"]
        assert distance.off_chart[0].conservative

    def test_two_substitutions_on_one_reading_are_both_reported(self):
        distance = perf.takeoff_distance(2550.0, -300.0, 20.0, headwind_kt=45.0)
        assert {e.what for e in distance.off_chart} == {
            "pressure altitude",
            "headwind",
        }

    def test_an_equal_density_cruise_reading_is_reported_as_optimistic(self):
        # ISA+35 at 4000 ft is off the right of the chart's temperature band.
        lookup = perf.cruise_at_density(4000.0, 2400.0, 42.0)
        assert lookup.substituted
        entry = lookup.off_chart[0]
        assert entry.what == "cruise temperature"
        assert not entry.conservative
        # It describes the query, not the page it escaped to.
        assert "at 4000 ft" in entry.detail

    def test_an_in_band_cruise_reading_claims_nothing(self):
        assert perf.cruise_at_density(4000.0, 2400.0, 7.0).off_chart == ()


# --- and the runway check carries it -------------------------------------


class TestTheRunwayCheckCarriesIt:
    def test_an_ordinary_field_is_all_from_the_book(self):
        result = check([pf.Runway("09/27", 5000.0, "ASPH")])
        assert not result.extrapolated
        assert result.off_chart == ()

    def test_a_pressure_altitude_below_the_chart_reaches_the_runway_row(self):
        result = check(
            [pf.Runway("09/27", 5000.0, "ASPH")], pressure_altitude_ft=-300.0
        )
        runway = result.runways[0]
        assert runway.extrapolated
        assert not runway.optimistic
        # Still a pass: the substitution errs long, so the runway check is if
        # anything harder than the truth.
        assert runway.passes is True

    def test_the_field_dedupes_a_record_every_runway_shares(self):
        # The pressure altitude is the field's, so all four ends produce the
        # identical record and listing it four times would bury the rest.
        result = check(
            [
                pf.Runway("09/27", 5000.0, "ASPH"),
                pf.Runway("18/36", 4000.0, "ASPH"),
            ],
            pressure_altitude_ft=-300.0,
        )
        assert len(result.off_chart) == 1


# --- and the verdict reflects it -----------------------------------------


class TestTheVerdict:
    def test_a_clean_plan_reads_go(self):
        result = pf.summarise([check([pf.Runway("09/27", 5000.0, "ASPH")])], fuel())
        assert result.is_go
        assert result.all_from_the_book
        assert result.verdict == "GO"
        assert result.extrapolations == ()

    def test_an_extrapolation_qualifies_the_word_without_failing_it(self):
        result = pf.summarise(
            [check([pf.Runway("09/27", 5000.0, "ASPH")], pressure_altitude_ft=-300.0)],
            fuel(),
        )
        # Still a go. A high-pressure morning at a sea-level field is an
        # ordinary day, and "NO GO" here would teach a pilot to ignore it.
        assert result.is_go
        assert not result.all_from_the_book
        assert result.verdict == "GO -- EXTRAPOLATED"
        assert len(result.extrapolations) == 1

    def test_a_real_failure_still_outranks_an_extrapolation(self):
        result = pf.summarise(
            [check([pf.Runway("09/27", 700.0, "ASPH")], pressure_altitude_ft=-300.0)],
            fuel(),
        )
        assert result.verdict == "NO GO"
        assert result.blockers
        assert result.extrapolations  # still listed, still not the reason

    def test_the_optimistic_ones_are_separable(self):
        conservative = perf.OffChart("pressure altitude", "read low", True)
        optimistic = perf.OffChart("cruise altitude", "read high", False)
        result = pf.summarise(
            [check([pf.Runway("09/27", 5000.0, "ASPH")])],
            fuel(),
            off_chart=[("leg 2", conservative), ("leg 5", optimistic)],
        )
        assert len(result.extrapolations) == 2
        assert [e.where for e in result.optimistic_extrapolations] == ["leg 5"]
        # Conservative first: the list is read top down and should end on the
        # one worth stopping for.
        assert result.extrapolations[-1].conservative is False

    def test_the_text_checklist_prints_them_under_the_verdict(self):
        result = pf.summarise(
            [check([pf.Runway("09/27", 5000.0, "ASPH")], pressure_altitude_ft=-300.0)],
            fuel(),
        )
        text = pf.format_checklist(result)
        assert text.splitlines()[0] == "GO / NO-GO: GO -- EXTRAPOLATED"
        assert any(line.startswith("EXTRAPOLATED:") for line in text.splitlines())


# --- and the navlog names the first row ----------------------------------


def hot_route(oat_c=38.0, altimeter_inhg=30.25, cruise_ft=3500.0):
    departure = nl.Waypoint(
        "KSQL", LatLon(37.512, -122.250), kind="airport", elevation_ft=5.0,
        runways=(pf.Runway("12/30", 2600.0, "ASPH"),),
        altimeter_inhg=altimeter_inhg, oat_c=oat_c,
    )
    destination = nl.Waypoint(
        "KMRY", LatLon(36.587, -121.843), kind="airport", elevation_ft=257.0,
        runways=(pf.Runway("10R/28L", 7616.0, "ASPH"),),
        altimeter_inhg=altimeter_inhg, oat_c=oat_c,
    )
    return nl.build_navlog(
        [departure, destination],
        cruise_ft,
        nl.Aircraft(weight_lb=2400.0, cruise_rpm=2400.0, fuel_on_board_gal=40.0),
        nl.Conditions(
            flight_date=date(2026, 9, 23),
            altimeter_inhg=altimeter_inhg,
            isa_deviation_c=20.0,
        ),
        planning_mode="auto",
    )


class TestTheNavlogNamesTheRow:
    def test_a_standard_day_marks_no_row(self):
        log = hot_route(oat_c=15.0, altimeter_inhg=29.92)
        assert log.extrapolated_legs == ()
        assert log.first_extrapolated_leg is None
        assert log.checklist.verdict == "GO"

    def test_a_hot_day_marks_the_rows_it_substituted_on(self):
        log = hot_route()
        assert log.extrapolated_legs
        assert all(log.legs[i - 1].extrapolated for i in log.extrapolated_legs)

    def test_the_first_such_row_is_the_one_reported(self):
        log = hot_route()
        assert log.first_extrapolated_leg == log.extrapolated_legs[0]

    def test_the_checklist_names_that_row(self):
        log = hot_route()
        first = log.first_extrapolated_leg
        assert any(
            e.where.startswith(f"leg {first} ")
            for e in log.checklist.extrapolations
        )

    def test_repeated_rows_are_grouped_behind_the_first(self):
        """One sentence per distinct reading, not one per row.

        A hot afternoon puts the same substitution on every level row, and
        eleven copies of it would bury the verdict it is qualifying.
        """
        log = hot_route()
        cruise_entries = [
            e for e in log.checklist.extrapolations if e.what == "cruise temperature"
        ]
        assert len(cruise_entries) == 1
        if len(log.extrapolated_legs) > 1:
            assert "later row" in cruise_entries[0].where

    def test_the_printed_log_marks_the_row_and_says_where_it_starts(self):
        log = hot_route()
        text = nl.format_navlog(log)
        assert "+ marks a row whose performance was extrapolated" in text
        assert f"The first is row {log.first_extrapolated_leg}" in text

    def test_the_verdict_is_qualified_rather_than_failed(self):
        log = hot_route()
        assert log.checklist.is_go
        assert log.checklist.verdict == "GO -- EXTRAPOLATED"


# --- the cruise band is the rule, so it gets its own tests ---------------


class TestTheCruiseBand:
    """ISA +/-20 at the queried pressure altitude. Outside it is extrapolation.

    The chart prints three columns per altitude page -- ISA-20, ISA, ISA+20 --
    and inside that band a reading is the chart's own. Outside it there is no
    column for the operating point, and `cruise_at_density` answers from air
    of the same density found elsewhere on the chart. That answer is an
    extrapolation of this aeroplane's published cruise performance, whatever
    the mechanism used to reach it.
    """

    def test_the_band_comes_from_the_digitized_chart(self):
        assert perf.cruise_isa_band() == (-20.0, 20.0)

    def test_the_line_sits_exactly_at_the_chart_edge(self):
        # Zero tolerance: no slop is allowed past the printed columns.
        assert perf.CRUISE_ISA_TOLERANCE_C == 0.0
        assert perf.cruise_extrapolation_band() == perf.cruise_isa_band()

    @pytest.mark.parametrize("isa_dev", [-20.0, -10.0, 0.0, 12.0, 20.0])
    def test_inside_the_band_is_the_chart_itself(self, isa_dev):
        from engine.atmosphere import isa_temperature_c

        lookup = perf.cruise_at_density(
            6000.0, 2400.0, isa_temperature_c(6000.0) + isa_dev
        )
        assert not lookup.substituted
        assert not lookup.extrapolated
        assert lookup.off_chart == ()
        # And it was read where it was asked, not beside it.
        assert lookup.pressure_altitude_ft == pytest.approx(6000.0)

    @pytest.mark.parametrize("isa_dev", [-30.0, -21.0, 21.0, 25.0, 35.0])
    def test_outside_the_band_is_an_extrapolation(self, isa_dev):
        from engine.atmosphere import isa_temperature_c

        lookup = perf.cruise_at_density(
            6000.0, 2400.0, isa_temperature_c(6000.0) + isa_dev
        )
        assert lookup.substituted
        assert lookup.extrapolated
        assert lookup.off_chart
        assert not lookup.off_chart[0].conservative
        # It names the query, the band, and where it actually read.
        detail = lookup.off_chart[0].detail
        assert "ISA-20 to ISA+20" in detail
        assert "6000 ft" in detail

    def test_one_degree_past_the_column_already_counts(self):
        """No grace band. ISA+21 is outside the chart and says so."""
        from engine.atmosphere import isa_temperature_c

        assert perf.cruise_at_density(
            4000.0, 2400.0, isa_temperature_c(4000.0) + 20.0
        ).extrapolated is False
        assert perf.cruise_at_density(
            4000.0, 2400.0, isa_temperature_c(4000.0) + 21.0
        ).extrapolated is True

    def test_the_band_is_measured_at_the_queried_pressure_altitude(self):
        """ISA is a different temperature at every altitude.

        ISA+20 is 25 C at 2000 ft and 13 C at 8000 ft. A band written in
        absolute degrees would be wrong at one end or the other; this one is
        anchored to the standard temperature of the altitude asked about.
        """
        from engine.atmosphere import isa_temperature_c

        for altitude in (2000.0, 8000.0, 12000.0):
            edge = isa_temperature_c(altitude) + 20.0
            assert not perf.cruise_at_density(altitude, 2400.0, edge).extrapolated

            # Just past it is never a silent pass. It is an extrapolation, or
            # -- high up, where the equal-density search runs off the top of
            # the chart -- a refusal, which is the stricter of the two.
            just_past = isa_temperature_c(altitude) + 20.5
            try:
                assert perf.cruise_at_density(
                    altitude, 2400.0, just_past
                ).extrapolated
            except perf.OutsidePOHEnvelope:
                pass

    def test_a_leg_flown_outside_the_band_marks_its_row(self):
        log = hot_route()
        cruise = next(leg for leg in log.legs if leg.phase == "cruise")
        assert cruise.extrapolated
        assert any(e.what == "cruise temperature" for e in cruise.off_chart)
