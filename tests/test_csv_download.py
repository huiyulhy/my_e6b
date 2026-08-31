"""Paper-nav-log CSV export tests.

The format's job is to line up with a printed pad, so what is tested is the
column order, the two inverted sign conventions, and which cells are left for
the pilot to fill in the aircraft.
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


class TestShape:
    def test_the_header_is_the_pads_column_order(self):
        text = cd.navlog_csv(log())
        assert text.splitlines()[0] == ",".join(cd.COLUMNS)

    def test_one_row_per_leg_plus_a_totals_row(self):
        navlog = log()
        assert len(rows(navlog)) == len(navlog.legs) + 1

    def test_the_totals_row_carries_the_totals(self):
        navlog = log()
        totals = rows(navlog)[-1]
        assert totals["Check Point"] == "Totals"
        assert float(totals["Dist Leg"]) == pytest.approx(
            navlog.total_distance_nm, abs=0.05
        )
        assert float(totals["ETE"]) == pytest.approx(navlog.total_time_min, abs=0.05)
        assert float(totals["Fuel"]) == pytest.approx(navlog.total_fuel_gal, abs=0.05)

    def test_a_row_is_named_for_where_its_leg_ends(self):
        """The pad's rows are checkpoints; the leg arriving there is written on it."""
        climb = next(r for r in rows(log()) if r["Phase"] == "climb")
        assert climb["Check Point"] == "TOC"

    def test_the_last_row_is_the_destination(self):
        body = [r for r in rows(log()) if r["Check Point"] != "Totals"]
        assert body[-1]["Check Point"] == "KSBP"


class TestConventions:
    def test_variation_is_east_positive_as_everywhere_else(self):
        """One number means one thing: `+12.8` is 12.8 east, subtracted from TC.

        A printed pad heads its column `-E / +W` and wants the opposite sign.
        This is the one column the export deliberately does not follow the pad
        on, rather than have the file disagree with the screen.
        """
        navlog = log()
        leg = next(leg for leg in navlog.legs if leg.phase == "cruise")
        row = flying(navlog)[0]
        assert float(row["Var"]) == pytest.approx(leg.variation_deg, abs=0.05)
        # California is easterly variation, so the column reads positive.
        assert leg.variation_deg > 0
        assert row["Var"].startswith("+")
        # And it is the number that turns this row's true course into its
        # magnetic heading, east being least.
        assert float(row["TH"]) - float(row["Var"]) == pytest.approx(
            float(row["MH"]), abs=1.0
        )

    def test_the_wind_correction_column_keeps_our_sign(self):
        """Both are `-L / +R`, so this one passes through."""
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

    def test_corrections_always_show_their_sign(self):
        for row in flying(log()):
            assert row["WCA"][0] in "+-"
            assert row["Var"][0] in "+-"

    def test_cas_is_below_tas_at_altitude(self):
        """The one derived airspeed: what the ASI reads for this row's TAS."""
        row = flying(log())[0]
        assert float(row["CAS"]) < float(row["TAS"])


class TestWhatIsLeftBlank:
    def test_the_columns_nothing_can_fill_are_empty(self):
        for row in rows(log()):
            for column in cd.BLANK_COLUMNS:
                assert row[column] == "", column

    def test_a_row_that_goes_nowhere_carries_only_its_fuel_and_time(self):
        """Taxi and the pattern: no course, no heading, nothing to fly."""
        for row in rows(log()):
            if row["Phase"] not in ("taxi", "pattern"):
                continue
            assert row["TC"] == "" and row["MH"] == "" and row["GS Est"] == ""
            assert float(row["Fuel"]) > 0
            assert row["Altitude"] != ""


class TestTimeOff:
    def test_no_departure_time_leaves_the_eta_blank(self):
        assert all(row["ETA"] == "" for row in rows(log()))

    def test_a_departure_time_fills_the_eta_column(self):
        navlog = log()
        body = [r for r in rows(navlog, time_off="13:45") if r["Check Point"] != "Totals"]
        assert body[0]["ETA"] == "13:45"  # the taxi row, before anything is flown
        assert body[-1]["ETA"] != ""

    def test_the_eta_is_the_departure_plus_the_time_so_far(self):
        navlog = log()
        body = [r for r in rows(navlog, time_off="13:45") if r["Check Point"] != "Totals"]
        last = navlog.legs[-1]
        expected = (13 * 60 + 45 + int(last.cumulative_ete_min)) % (24 * 60)
        hh, mm = (int(p) for p in body[-1]["ETA"].split(":"))
        assert abs((hh * 60 + mm) - expected) <= 1

    def test_the_eta_column_agrees_with_the_ete_column(self):
        """A pilot adding the ETE column down the page must land on the ETA.

        Minutes are rounded, not truncated: a 12.99-minute leg prints 13.0, and
        an ETA a minute early would be a plan that disagrees with itself.
        """
        navlog = log()
        body = [r for r in rows(navlog, time_off="13:45") if r["Check Point"] != "Totals"]
        running = 0.0
        for row in body:
            running += float(row["ETE"])
            hh, mm = (int(p) for p in row["ETA"].split(":"))
            assert (hh * 60 + mm) == round(13 * 60 + 45 + running), row["Check Point"]

    def test_it_wraps_past_midnight(self):
        assert cd._clock(cd._parse_time_off("23:30"), 60.0) == "00:30"

    def test_a_bare_four_digit_time_is_accepted(self):
        """Because that is how a pilot writes one."""
        assert cd._parse_time_off("1345") == cd._parse_time_off("13:45")

    def test_a_time_that_is_not_a_time_is_refused(self):
        with pytest.raises(ValueError, match="24-hour time"):
            cd._parse_time_off("quarter to two")


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
        navlog = nl.build_navlog(
            [KSQL, odd], 4500, conditions=CALM, planning_mode="auto"
        )
        assert any(
            row["Check Point"] == "HALF MOON, CA" for row in rows(navlog)
        ), cd.navlog_csv(navlog)
