"""Navlog consistency tests.

One test per implemented check, plus a clean plan that must come back
consistent. The stubs are covered only by `test_stubs_return_nothing_yet`,
which exists so that filling one in fails loudly here rather than silently
changing behaviour somewhere else.
"""

from dataclasses import replace
from datetime import date

import pytest

from engine import consistency as cy
from engine import navlog as nl
from engine.geo import LatLon

KSQL = nl.Waypoint("KSQL", LatLon(37.5119, -122.2495), "airport", elevation_ft=5)
KMRY = nl.Waypoint("KMRY", LatLon(36.5870, -121.8429), "airport", elevation_ft=257)
VPWDM = nl.Waypoint("VPWDM", LatLon(37.2000, -122.0500), "vfr_waypoint")

CALM = nl.Conditions(flight_date=date(2026, 8, 15))


def declare(waypoint, segment_type, altitude_ft=None):
    return replace(waypoint, segment_type=segment_type, altitude_ft=altitude_ft)


def codes(report):
    return {f.code for f in report.findings}


class TestCleanPlans:
    def test_an_automatic_plan_is_consistent(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        report = cy.check_navlog_consistency(log)
        assert report.is_consistent, [f.message for f in report.findings]

    def test_a_declared_plan_is_consistent(self):
        route = [KSQL, declare(VPWDM, "climb"), declare(KMRY, "descent")]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        report = cy.check_navlog_consistency(log)
        assert report.is_consistent, [f.message for f in report.findings]

    def test_a_clean_report_has_no_errors(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert cy.check_navlog_consistency(log).errors == ()


class TestDeclaredVsActual:
    def test_a_climb_that_descends_is_an_error(self):
        route = [
            KSQL,
            declare(VPWDM, "climb", 6500),
            declare(KMRY, "climb", 3000),  # declared a climb, but goes down
        ]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        report = cy.check_navlog_consistency(log)
        assert "declared-vs-actual" in codes(report)
        assert not report.is_consistent

    def test_a_cruise_that_changes_altitude_is_an_error(self):
        route = [
            KSQL,
            declare(VPWDM, "climb", 4500),
            declare(KMRY, "cruise", 2500),  # declared level, but descends
        ]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        assert "declared-vs-actual" in codes(cy.check_navlog_consistency(log))

    def test_the_message_names_the_leg_and_both_altitudes(self):
        route = [KSQL, declare(VPWDM, "climb", 6500), declare(KMRY, "climb", 3000)]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        finding = next(
            f for f in cy.check_navlog_consistency(log).findings if f.code == "declared-vs-actual"
        )
        assert "VPWDM" in finding.message
        assert "6500" in finding.message and "3000" in finding.message
        assert finding.row is not None


class TestAltitudeContinuity:
    def test_a_gap_between_legs_is_an_error(self):
        """Built by hand: the planner cannot produce one, which is the point."""
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        flown = [i for i, leg in enumerate(log.legs) if leg.covers_ground]
        broken = list(log.legs)
        first = broken[flown[0]]
        broken[flown[0]] = replace(first, exit_altitude_ft=(first.exit_altitude_ft or 0) + 2000)
        report = cy.check_navlog_consistency(replace(log, legs=tuple(broken)))
        assert "altitude-discontinuity" in codes(report)
        assert not report.is_consistent

    def test_a_continuous_profile_passes(self):
        log = nl.build_navlog([KSQL, VPWDM, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert "altitude-discontinuity" not in codes(cy.check_navlog_consistency(log))


class TestArrivalAltitude:
    def test_ending_the_flight_in_a_climb_is_flagged(self):
        route = [KSQL, declare(VPWDM, "cruise", 3000), declare(KMRY, "climb", 6500)]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        report = cy.check_navlog_consistency(log)
        assert "arrival-altitude" in codes(report)
        # A warning, not an error -- odd, but not self-contradictory.
        assert all(f.severity == "warning" for f in report.findings if f.code == "arrival-altitude")

    def test_a_normal_arrival_is_not_flagged(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert "arrival-altitude" not in codes(cy.check_navlog_consistency(log))


class TestStaleOverrides:
    def test_an_edit_on_a_ground_row_is_flagged(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        taxi = next(i for i, leg in enumerate(log.legs) if leg.phase == "taxi")
        rows = list(log.legs)
        rows[taxi] = replace(rows[taxi], overridden=("wind_from_deg",))
        report = cy.check_navlog_consistency(replace(log, legs=tuple(rows)))
        assert "stale-override" in codes(report)


class TestClimbAchievable:
    """A short leg cannot be given a top the POH will not deliver."""

    def test_a_climb_beyond_the_leg_is_an_error(self):
        # KSQL to VPWDM is about 20 nm; a 172 does not get to 11500 ft in that.
        route = [KSQL, declare(VPWDM, "climb", 11500), declare(KMRY, "descent", 257)]
        log = nl.build_navlog(route, 11500, conditions=CALM, planning_mode="manual")
        report = cy.check_navlog_consistency(log)
        assert "climb-unreachable" in codes(report)
        assert not report.is_consistent

    def test_the_message_names_the_leg_and_what_is_reachable(self):
        route = [KSQL, declare(VPWDM, "climb", 11500), declare(KMRY, "descent", 257)]
        log = nl.build_navlog(route, 11500, conditions=CALM, planning_mode="manual")
        finding = next(
            f for f in cy.check_navlog_consistency(log).findings if f.code == "climb-unreachable"
        )
        assert "VPWDM" in finding.message
        assert "11500" in finding.message
        assert finding.row is not None

    def test_an_automatic_climb_is_never_unreachable(self):
        """Auto mode stops the climb where the aeroplane stops; nothing to flag."""
        log = nl.build_navlog([KSQL, KMRY], 8500, conditions=CALM, planning_mode="auto")
        assert "climb-unreachable" not in codes(cy.check_navlog_consistency(log))

    def test_a_climb_the_leg_can_just_about_make_passes(self):
        log = nl.build_navlog([KSQL, VPWDM, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert "climb-unreachable" not in codes(cy.check_navlog_consistency(log))


class TestDescentRate:
    def test_an_automatic_descent_is_not_flagged(self):
        """The regression the epsilon exists for: auto descends at exactly 500."""
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        assert "descent-too-steep" not in codes(cy.check_navlog_consistency(log))

    def test_a_descent_steeper_than_the_limit_is_an_error(self):
        # VPWDM to KMRY is about 45 nm, and 9500 ft to the field in half of it
        # needs well over 500 ft/min.
        near_kmry = nl.Waypoint("VPTOP", LatLon(36.7500, -121.9000), "vfr_waypoint")
        route = [
            KSQL,
            declare(VPWDM, "climb", 9500),
            declare(near_kmry, "cruise", 9500),
            declare(KMRY, "descent", 257),
        ]
        log = nl.build_navlog(route, 9500, conditions=CALM, planning_mode="manual")
        report = cy.check_navlog_consistency(log)
        assert "descent-too-steep" in codes(report)
        assert not report.is_consistent

    def test_the_message_gives_the_rate(self):
        near_kmry = nl.Waypoint("VPTOP", LatLon(36.7500, -121.9000), "vfr_waypoint")
        route = [
            KSQL,
            declare(VPWDM, "climb", 9500),
            declare(near_kmry, "cruise", 9500),
            declare(KMRY, "descent", 257),
        ]
        log = nl.build_navlog(route, 9500, conditions=CALM, planning_mode="manual")
        finding = next(
            f for f in cy.check_navlog_consistency(log).findings if f.code == "descent-too-steep"
        )
        assert "ft/min" in finding.message
        assert "KMRY" in finding.message


class TestVFRHemispheric:
    """FAR 91.159, applied on MSL because the engine has no terrain."""

    # KSQL to KMRY runs south-southeast: a magnetic course in the 100s, so the
    # eastbound half of the rule, odd thousands plus 500.
    def test_an_eastbound_cruise_at_an_even_altitude_is_flagged(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        report = cy.check_navlog_consistency(log)
        assert "vfr-hemispheric" in codes(report)
        assert report.is_consistent  # a warning, not a contradiction

    def test_an_eastbound_cruise_at_an_odd_altitude_passes(self):
        log = nl.build_navlog([KSQL, KMRY], 7500, conditions=CALM, planning_mode="auto")
        assert "vfr-hemispheric" not in codes(cy.check_navlog_consistency(log))

    def test_a_westbound_cruise_at_an_even_altitude_passes(self):
        log = nl.build_navlog([KMRY, KSQL], 6500, conditions=CALM, planning_mode="auto")
        assert "vfr-hemispheric" not in codes(cy.check_navlog_consistency(log))

    def test_a_westbound_cruise_at_an_odd_altitude_is_flagged(self):
        log = nl.build_navlog([KMRY, KSQL], 7500, conditions=CALM, planning_mode="auto")
        assert "vfr-hemispheric" in codes(cy.check_navlog_consistency(log))

    def test_a_round_thousand_is_flagged(self):
        log = nl.build_navlog([KSQL, KMRY], 7000, conditions=CALM, planning_mode="auto")
        assert "vfr-hemispheric" in codes(cy.check_navlog_consistency(log))

    def test_at_or_below_3000_is_not_asked(self):
        log = nl.build_navlog([KSQL, KMRY], 3000, conditions=CALM, planning_mode="auto")
        assert "vfr-hemispheric" not in codes(cy.check_navlog_consistency(log))

    def test_only_level_flight_is_asked(self):
        """The climb and descent rows pass through even altitudes untouched."""
        log = nl.build_navlog([KSQL, KMRY], 7500, conditions=CALM, planning_mode="auto")
        report = cy.check_navlog_consistency(log)
        assert not [f for f in report.findings if f.code == "vfr-hemispheric"]

    def test_the_message_names_the_nearest_legal_altitude(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        finding = next(
            f for f in cy.check_navlog_consistency(log).findings if f.code == "vfr-hemispheric"
        )
        assert "91.159" in finding.message
        assert "6500" in finding.message
        assert "odd" in finding.message

    @pytest.mark.parametrize(
        ("altitude", "wants_odd", "expected"),
        [
            (7500, True, 7500),
            (6500, True, 5500),  # equidistant from 5500 and 7500; ties go down
            (7000, True, 7500),  # a round thousand is 500 from the one above
            (6000, False, 6500),
            (5500, False, 4500),  # odd, westbound: tie again, and down again
        ],
    )
    def test_the_nearest_legal_altitude(self, altitude, wants_odd, expected):
        assert cy._legal_vfr_altitude(altitude, wants_odd=wants_odd) == expected


class TestReportShape:
    def test_findings_are_sorted_most_severe_first(self):
        route = [KSQL, declare(VPWDM, "climb", 6500), declare(KMRY, "climb", 3000)]
        log = nl.build_navlog(route, 7500, conditions=CALM, planning_mode="manual")
        report = cy.check_navlog_consistency(log)
        ranks = [cy.SEVERITIES.index(f.severity) for f in report.findings]
        assert ranks == sorted(ranks)

    def test_warnings_alone_do_not_make_a_plan_inconsistent(self):
        report = cy.ConsistencyReport(
            findings=(cy.Finding(severity="warning", code="x", message="m"),)
        )
        assert report.is_consistent

    def test_an_error_does(self):
        report = cy.ConsistencyReport(
            findings=(cy.Finding(severity="error", code="x", message="m"),)
        )
        assert not report.is_consistent

    def test_an_unknown_severity_is_refused(self):
        with pytest.raises(ValueError, match="severity"):
            cy.Finding(severity="catastrophic", code="x", message="m")

    def test_format_report_says_so_when_there_is_nothing(self):
        assert "no problems" in cy.format_report(cy.ConsistencyReport(findings=()))

    def test_format_report_lists_each_finding(self):
        report = cy.ConsistencyReport(
            findings=(cy.Finding(severity="error", code="x", message="the thing", row=2),)
        )
        text = cy.format_report(report)
        assert "the thing" in text
        assert "row 3" in text  # one-based for the pilot


class TestStubsAreRegisteredButEmpty:
    """The unimplemented checks are wired in and return nothing.

    They are registered on purpose: filling one in should be a one-function
    change, and the list of what is *not* yet checked should be visible in the
    code rather than remembered.
    """

    STUBS = (
        cy._check_toc_before_tod,
        cy._check_cruise_below_chart,
    )

    def test_all_stubs_are_registered(self):
        for stub in self.STUBS:
            assert stub in cy._CHECKS

    def test_stubs_return_nothing_yet(self):
        log = nl.build_navlog([KSQL, KMRY], 6500, conditions=CALM, planning_mode="auto")
        ctx = cy._Context(navlog=log, aircraft=nl.Aircraft(), conditions=CALM)
        for stub in self.STUBS:
            assert stub(ctx) == [], f"{stub.__name__} is implemented; give it a test"
