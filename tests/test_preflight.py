"""The go/no-go checklist, and in particular what the wind does to it.

Distances themselves are `tests/test_performance.py`'s job. What is tested
here is the part the checklist adds on top: which end of which runway the
numbers are read for, and which of length and crosswind decided the verdict.
"""

from __future__ import annotations

import pytest

from engine import performance as perf
from engine import preflight as pf

# A field where the book numbers are comfortable, so the wind is the only
# thing moving the answer.
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


# --- reading a designator ------------------------------------------------


@pytest.mark.parametrize(
    "designation, expected",
    [
        ("12/30", [("12", 120.0), ("30", 300.0)]),
        ("09L/27R", [("09L", 90.0), ("27R", 270.0)]),
        ("1/19", [("1", 10.0), ("19", 190.0)]),
        ("36", [("36", 360.0)]),
    ],
)
def test_ends_come_off_the_designator(designation, expected):
    ends = pf.Runway(designation, 3000.0).ends
    assert [(e.label, e.magnetic_heading_deg) for e in ends] == expected


@pytest.mark.parametrize("designation", ["", "H1", "N/S", "ALL/WAY", "0/40"])
def test_undecipherable_designators_give_no_ends(designation):
    assert pf.Runway(designation, 3000.0).ends == ()


# --- resolving a wind onto an end ---------------------------------------


def test_straight_down_the_runway_is_all_headwind():
    got = pf.wind_components(90.0, pf.SurfaceWind(from_deg=90.0, speed_kt=12.0))
    assert got.headwind_kt == pytest.approx(12.0)
    assert got.crosswind_kt == pytest.approx(0.0)


def test_ninety_degrees_off_is_all_crosswind():
    got = pf.wind_components(360.0, pf.SurfaceWind(from_deg=90.0, speed_kt=12.0))
    assert got.headwind_kt == pytest.approx(0.0, abs=1e-9)
    assert got.crosswind_kt == pytest.approx(12.0)
    assert got.crosswind_from_right


def test_a_wind_from_behind_reads_as_a_negative_headwind():
    got = pf.wind_components(360.0, pf.SurfaceWind(from_deg=180.0, speed_kt=8.0))
    assert got.headwind_kt == pytest.approx(-8.0)
    assert got.is_tailwind


def test_the_gust_counts_against_you_and_never_for_you():
    wind = pf.SurfaceWind(from_deg=45.0, speed_kt=10.0, gust_kt=20.0)
    # Into wind the steady figure is the one the roll gets.
    into = pf.wind_components(45.0, wind)
    assert into.headwind_kt == pytest.approx(10.0)
    # Downwind the gust is the one that might be there when it matters.
    away = pf.wind_components(225.0, wind)
    assert away.headwind_kt == pytest.approx(-20.0)
    # And the crosswind is always taken at the peak.
    across = pf.wind_components(135.0, wind)
    assert across.crosswind_kt == pytest.approx(20.0)


def test_a_gust_below_the_steady_wind_is_refused():
    with pytest.raises(ValueError):
        pf.SurfaceWind(from_deg=45.0, speed_kt=15.0, gust_kt=10.0)


# --- the wind reaching the distances ------------------------------------


def test_the_favourable_end_is_the_one_chosen():
    result = check(
        [pf.Runway("12/30", 3000.0, "ASPH")],
        pf.SurfaceWind(from_deg=300.0, speed_kt=15.0),
    )
    runway = result.runways[0]
    assert runway.end_used == "30"
    assert runway.headwind_kt == pytest.approx(15.0)


def test_a_headwind_shortens_the_roll_and_a_tailwind_lengthens_it():
    calm = check([pf.Runway("09/27", 5000.0, "ASPH")]).runways[0]
    into = check(
        [pf.Runway("09/27", 5000.0, "ASPH")],
        pf.SurfaceWind(from_deg=90.0, speed_kt=18.0),
    ).runways[0]
    # One runway, one wind, so a tailwind only happens where the wind is
    # square across the field and the better end still has one -- which needs
    # a single-ended designator.
    behind = check(
        [pf.Runway("09", 5000.0, "ASPH")],
        pf.SurfaceWind(from_deg=270.0, speed_kt=8.0),
    ).runways[0]

    assert into.ground_roll_ft < calm.ground_roll_ft
    assert behind.ground_roll_ft > calm.ground_roll_ft


def test_no_wind_given_leaves_the_book_figures_and_says_so():
    result = check([pf.Runway("09/27", 5000.0, "ASPH")])
    runway = result.runways[0]
    assert runway.headwind_kt is None
    assert runway.end_used == ""
    assert "no surface wind given" in runway.note
    assert result.passes is True


def test_a_wind_with_no_heading_to_resolve_it_onto_is_reported():
    result = check(
        [pf.Runway("H1", 5000.0, "ASPH")],
        pf.SurfaceWind(from_deg=90.0, speed_kt=10.0),
    )
    assert "no heading" in result.runways[0].note
    assert result.runways[0].headwind_kt is None


# --- the crosswind gate --------------------------------------------------


def test_a_crosswind_past_the_demonstrated_maximum_fails_a_long_runway():
    # Square across a 10,000 ft runway: length is not the problem.
    over = perf.MAX_DEMONSTRATED_CROSSWIND_KT + 10.0
    result = check(
        [pf.Runway("18/36", 10000.0, "ASPH")],
        pf.SurfaceWind(from_deg=90.0, speed_kt=over),
    )
    runway = result.runways[0]
    assert runway.crosswind_kt == pytest.approx(over)
    assert runway.crosswind_exceeds_demonstrated
    assert runway.passes is False
    assert result.passes is False
    assert runway.spare_ft > 0  # it was long enough; that was never the issue


def test_the_gust_is_what_puts_the_crosswind_over():
    steady = pf.SurfaceWind(from_deg=90.0, speed_kt=12.0)
    gusting = pf.SurfaceWind(from_deg=90.0, speed_kt=12.0, gust_kt=22.0)
    assert check([pf.Runway("18/36", 6000.0, "ASPH")], steady).passes is True
    assert check([pf.Runway("18/36", 6000.0, "ASPH")], gusting).passes is False


def test_one_usable_runway_is_enough():
    # Wind straight down 09/27 is straight across 18/36.
    result = check(
        [pf.Runway("18/36", 6000.0, "ASPH"), pf.Runway("09/27", 4000.0, "ASPH")],
        pf.SurfaceWind(from_deg=90.0, speed_kt=25.0),
    )
    assert result.passes is True
    assert result.best.runway == "09/27"  # not the longer one it cannot land on


def test_a_tailwind_past_the_chart_is_a_no_go_rather_than_an_extrapolation():
    result = check(
        [pf.Runway("09", 6000.0, "ASPH")],
        pf.SurfaceWind(
            from_deg=270.0, speed_kt=perf.MAX_CHART_TAILWIND_KT + 8.0
        ),
    )
    runway = result.runways[0]
    assert runway.outside_envelope
    assert runway.passes is False
    assert "tailwind" in runway.note


# --- what the verdict says -----------------------------------------------


def _fuel(passes=True):
    return pf.check_fuel(
        fuel_on_board_gal=50.0 if passes else 4.0,
        burn_gal=10.0,
        reserve_required_gal=4.0,
        reserve_minutes=30.0,
        margin=0.1,
        night=False,
    )


def test_the_blocker_names_the_crosswind_not_the_length():
    result = pf.summarise(
        [check([pf.Runway("18/36", 10000.0, "ASPH")],
               pf.SurfaceWind(from_deg=90.0, speed_kt=30.0))],
        _fuel(),
    )
    assert not result.is_go
    assert len(result.blockers) == 1
    assert "crosswind" in result.blockers[0]
    assert "long enough" not in result.blockers[0]


def test_the_blocker_still_names_the_length_when_that_is_what_failed():
    result = pf.summarise(
        [check([pf.Runway("09/27", 800.0, "ASPH")],
               pf.SurfaceWind(from_deg=90.0, speed_kt=6.0))],
        _fuel(),
    )
    assert "no runway is long enough" in result.blockers[0]


def test_the_text_checklist_shows_the_wind_it_used():
    result = pf.summarise(
        [check([pf.Runway("09/27", 5000.0, "ASPH")],
               pf.SurfaceWind(from_deg=80.0, speed_kt=12.0, gust_kt=18.0))],
        _fuel(),
    )
    text = pf.format_checklist(result)
    assert "wind 080M at 12G18 kt" in text
    # The end chosen and its components, on the runway's own row.
    row = next(line for line in text.splitlines() if line.startswith("  09/27"))
    assert row.split() == ["09/27", "ASPH", "09", "12", "3", "5000",
                           "834", "1418", "1702", "3298", "ok"]


def test_variation_moves_the_wind_onto_the_runway():
    """The conversion the caller owes: true in, magnetic out.

    Thirteen degrees of easterly variation turns a 283 true wind into 270
    magnetic, which is straight down 27 rather than 13 degrees off it.
    """
    from engine.magnetic import true_to_magnetic

    magnetic = true_to_magnetic(283.0, 13.0)
    got = pf.wind_components(270.0, pf.SurfaceWind(from_deg=magnetic, speed_kt=20.0))
    assert got.crosswind_kt == pytest.approx(0.0, abs=1e-9)
    assert got.headwind_kt == pytest.approx(20.0)


def test_a_field_where_each_runway_failed_differently_says_so():
    """A gale across the long runway and down the short one's tail.

    Neither failed on length, so neither should be reported as short.
    """
    result = pf.summarise(
        [
            check(
                [pf.Runway("18/36", 10000.0, "ASPH"), pf.Runway("09", 2000.0, "TURF")],
                pf.SurfaceWind(from_deg=270.0, speed_kt=26.0, gust_kt=34.0),
            )
        ],
        _fuel(),
    )
    blocker = result.blockers[0]
    assert "no usable runway" in blocker
    assert "crosswind" in blocker
    assert "tailwind" in blocker
    assert "long enough" not in blocker
