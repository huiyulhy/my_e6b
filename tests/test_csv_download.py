"""Navlog CSV export tests.

The format's job is to match the navlog on screen, row for row and column for
column, with Leg Start / Leg End in place of the positions. What is tested is
the column order, the conventions the screen uses, and which cells a row that
goes nowhere leaves empty.
"""

import csv
import io
from datetime import date

import pytest

from engine import csv_download as cd
from engine import navlog as nl
from engine.geo import LatLon

KSQL = nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5)
KSBP = nl.Waypoint("KSBP", LatLon(35.2368, -120.6424), "airport", elevation_ft=212)

CALM = nl.Conditions(flight_date=date(2026, 8, 15))


def log(**kwargs):
    return nl.build_navlog(
        [KSQL, KSBP], 7500, conditions=CALM, planning_mode="auto", **kwargs
    )


def rows(navlog, **kwargs):
    text = cd.navlog_csv(navlog, **kwargs)
    return list(csv.DictReader(io.StringIO(text)))


def flying(navlog, **kwargs):
    return [r for r in rows(navlog, **kwargs) if r["Phase"] == "cruise"]


def body(navlog, **kwargs):
    return [r for r in rows(navlog, **kwargs) if r["Leg Start"] != "Total"]


class TestShape:
    def test_the_header_is_the_screens_column_order(self):
        text = cd.navlog_csv(log())
        assert text.splitlines()[0] == ",".join(cd.COLUMNS)
        assert cd.COLUMNS[:4] == ("Leg Start", "Leg End", "Phase", "End Alt")
        assert "Lat" not in cd.COLUMNS and "Lon" not in cd.COLUMNS

    def test_one_row_per_navlog_row_plus_a_totals_row(self):
        navlog = log()
        assert len(rows(navlog)) == len(navlog.legs) + 1

    def test_the_totals_row_carries_the_totals(self):
        navlog = log()
        totals = rows(navlog)[-1]
        assert totals["Leg Start"] == "Total"
        assert float(totals["Dist"]) == pytest.approx(navlog.total_distance_nm, abs=0.05)
        assert float(totals["ETE"]) == pytest.approx(navlog.total_time_min, abs=0.05)
        assert float(totals["Fuel"]) == pytest.approx(navlog.total_fuel_gal, abs=0.05)
        assert float(totals["Rem"]) == pytest.approx(navlog.fuel_remaining_gal, abs=0.05)

    def test_a_row_runs_from_its_leg_start_to_its_leg_end(self):
        climb = next(r for r in rows(log()) if r["Phase"] == "climb")
        assert (climb["Leg Start"], climb["Leg End"]) == ("KSQL", "TOC")
        cruise = next(r for r in rows(log()) if r["Phase"] == "cruise")
        assert (cruise["Leg Start"], cruise["Leg End"]) == ("TOC", "TOD")

    def test_the_last_flown_row_ends_at_the_destination(self):
        flown = [r for r in body(log()) if r["Leg End"]]
        assert flown[-1]["Leg End"] == "KSBP"

    def test_end_alt_is_where_the_row_ends(self):
        navlog = log()
        for leg, row in zip(navlog.legs, rows(navlog), strict=False):
            if leg.covers_ground:
                assert float(row["End Alt"]) == pytest.approx(leg.exit_altitude_ft, abs=0.5)

    def test_it_matches_the_screens_numbers(self):
        """CAS, TAS, GS, the air: the same figures the table shows."""
        navlog = log()
        for leg, row in zip(navlog.legs, rows(navlog), strict=False):
            if not leg.covers_ground:
                continue
            assert float(row["TAS"]) == pytest.approx(leg.tas_kt, abs=0.5)
            assert float(row["GS"]) == pytest.approx(leg.ground_speed_kt, abs=0.5)
            assert float(row["DA"]) == pytest.approx(leg.density_altitude_ft, abs=0.5)
            assert float(row["Dist"]) == pytest.approx(leg.distance_nm, abs=0.05)


class TestConventions:
    def test_variation_is_east_positive_as_everywhere_else(self):
        """One number means one thing: `+12.8` is 12.8 east, subtracted from TC."""
        navlog = log()
        leg = next(leg for leg in navlog.legs if leg.phase == "cruise")
        row = flying(navlog)[0]
        assert float(row["Var"]) == pytest.approx(leg.variation_deg, abs=0.05)
        assert leg.variation_deg > 0
        assert row["Var"].startswith("+")
        assert float(row["TH"]) - float(row["Var"]) == pytest.approx(float(row["MH"]), abs=1.0)

    def test_the_wind_correction_column_keeps_our_sign(self):
        navlog = log()
        leg = next(leg for leg in navlog.legs if leg.phase == "cruise")
        assert float(flying(navlog)[0]["WCA"]) == pytest.approx(
            leg.wind_correction_angle_deg, abs=0.05
        )

    def test_bearings_keep_their_leading_zeros(self):
        assert cd._bearing(7.0) == "007"
        # Rounds to 360, which is 000 -- never a fourth character.
        assert cd._bearing(359.6) == "000"
        assert cd._bearing(0.0) == "000"
        for row in flying(log()):
            assert len(row["TC"]) == 3
            assert len(row["MH"]) == 3
            assert len(row["Wind Dir"]) == 3

    def test_corrections_always_show_their_sign(self):
        for row in flying(log()):
            assert row["WCA"][0] in "+-"
            assert row["Var"][0] in "+-"

    def test_cas_is_below_tas_at_altitude(self):
        row = flying(log())[0]
        assert float(row["CAS"]) < float(row["TAS"])


class TestRowsThatGoNowhere:
    def test_taxi_and_pattern_carry_time_fuel_and_air_only(self):
        for row in rows(log()):
            if row["Phase"] not in ("taxi", "pattern"):
                continue
            for column in cd.NAVIGATION_COLUMNS:
                assert row[column] == "", column
            assert row["Leg End"] == ""
            assert float(row["Fuel"]) > 0
            assert row["End Alt"] != "" and row["DA"] != ""


class TestTimeRemaining:
    """Minutes left after each row, to the end of the flight: it counts down
    the way the fuel Rem column does, and needs no departure time."""

    def test_it_counts_down_to_zero(self):
        lines = body(log())
        remaining = [float(r["Time Rem"]) for r in lines]
        assert remaining == sorted(remaining, reverse=True)
        assert remaining[-1] == pytest.approx(0.0, abs=0.05)

    def test_it_starts_at_the_whole_flight(self):
        navlog = log()
        # After the taxi row, which takes no time on the log, all of it is left.
        assert float(body(navlog)[0]["Time Rem"]) == pytest.approx(navlog.total_time_min, abs=0.05)

    def test_it_is_the_total_less_the_time_so_far(self):
        navlog = log()
        for leg, row in zip(navlog.legs, body(navlog), strict=True):
            assert float(row["Time Rem"]) == pytest.approx(
                navlog.total_time_min - leg.cumulative_ete_min, abs=0.05
            )

    def test_the_pattern_is_what_is_left_on_arrival(self):
        navlog = log()
        arriving = [r for r in body(navlog) if r["Leg End"]][-1]
        assert float(arriving["Time Rem"]) == pytest.approx(navlog.legs[-1].ete_min, abs=0.05)

    def test_the_totals_row_has_none(self):
        assert rows(log())[-1]["Time Rem"] == ""


class TestFilename:
    def test_it_names_the_route(self):
        assert cd.csv_filename(log()) == "navlog-KSQL-KSBP.csv"

    def test_it_strips_anything_a_filesystem_would_mind(self):
        assert cd._safe("KSQL/../etc") == "KSQLetc"


class TestQuoting:
    def test_a_name_with_a_comma_survives_the_round_trip(self):
        odd = nl.Waypoint(
            "HALF MOON, CA", LatLon(37.5133, -122.5011), "airport", elevation_ft=66
        )
        navlog = nl.build_navlog([KSQL, odd], 4500, conditions=CALM, planning_mode="auto")
        assert any(
            row["Leg End"] == "HALF MOON, CA" for row in rows(navlog)
        ), cd.navlog_csv(navlog)
