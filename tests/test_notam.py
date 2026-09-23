"""Which NOTAMs are about this flight, and which are not.

The filter's job is to *reject*, so most of what is tested here is rejection:
the crane off track, the airway closure at FL300, the runway shut next
Tuesday. The other half is the rule that keeps it honest -- a NOTAM that does
not say where it is, or how high, or when, is never rejected on that ground.

Rejecting wrongly is the failure that matters. A NOTAM shown that need not
have been is a line a pilot skims; a NOTAM hidden that mattered is the closed
runway they land on.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from engine import notam as nt
from engine.geo import LatLon, inverse

# A leg running due east along 37N, from 122W to 121W -- about 48 nm.
WEST = LatLon(37.0, -122.0)
EAST = LatLon(37.0, -121.0)
SPAN = inverse(WEST, EAST)

NOON = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
ONE = datetime(2026, 9, 4, 13, 0, tzinfo=UTC)


def leg(lower_ft=6500.0, upper_ft=6500.0, start=NOON, end=ONE, span=SPAN):
    return nt.RouteWindow(
        span=span, lower_ft=lower_ft, upper_ft=upper_ft, start=start, end=end
    )


def notam(**fields):
    base = {
        "key": fields.pop("key", "N1"),
        "number": "01/001",
        "text": "TWY A CLSD",
        "position": SPAN.point_at_fraction(0.5),
        "radius_nm": 0.0,
        "lower_ft": 0.0,
        "upper_ft": 10000.0,
        "effective_start": NOON,
        "effective_end": ONE,
    }
    return nt.Notam(**{**base, **fields})


def keep(notams, window=None, **kwargs):
    return nt.relevant(notams, window or [leg()], **kwargs)


def off_track(nm):
    """A point that far north of the middle of the leg."""
    from engine.geo import direct

    return direct(SPAN.point_at_fraction(0.5), 0.0, nm)


class TestTheCorridor:
    def test_a_notam_on_the_track_is_kept(self):
        kept = keep([notam()])
        assert len(kept) == 1
        assert kept[0].distance_nm == pytest.approx(0.0, abs=0.1)
        assert "on the track" in kept[0].reasons[0]

    def test_one_just_inside_the_corridor_is_kept(self):
        kept = keep([notam(position=off_track(19.0))])
        assert len(kept) == 1
        assert kept[0].distance_nm == pytest.approx(19.0, abs=0.3)

    def test_one_just_outside_it_is_dropped(self):
        assert keep([notam(position=off_track(21.0))]) == ()

    def test_the_corridor_width_is_the_callers(self):
        far = [notam(position=off_track(30.0))]
        assert keep(far) == ()
        assert len(keep(far, corridor_nm=40.0)) == 1

    def test_a_notams_own_radius_reaches_into_the_corridor(self):
        """A five-mile circle eighteen miles off track touches a 20 nm
        corridor; a point at the same place is fifteen miles clear of it."""
        assert keep([notam(position=off_track(34.0))]) == ()
        kept = keep([notam(position=off_track(34.0), radius_nm=15.0)])
        assert len(kept) == 1
        assert kept[0].distance_nm == pytest.approx(19.0, abs=0.5)

    def test_distance_is_measured_to_the_leg_not_the_line_through_it(self):
        """A NOTAM two hundred miles beyond the destination is on the great
        circle through the leg, and is not near the leg."""
        from engine.geo import direct

        beyond = direct(EAST, SPAN.true_course_deg, 200.0)
        assert keep([notam(position=beyond)]) == ()

    def test_a_notam_with_no_position_is_kept_and_said_to_be_unplaced(self):
        """It cannot be ruled out on where it is, so it is not ruled out."""
        kept = keep([notam(position=None)])
        assert len(kept) == 1
        assert kept[0].distance_nm is None
        assert "no position" in kept[0].reasons[0]


class TestTheAltitudeBand:
    def test_an_airway_closure_far_above_is_dropped(self):
        """FL240 to FL350 is not about a Skyhawk at 6,500 ft."""
        assert keep([notam(lower_ft=24000.0, upper_ft=35000.0)]) == ()

    def test_a_band_that_reaches_the_cruise_altitude_is_kept(self):
        assert len(keep([notam(lower_ft=5000.0, upper_ft=8000.0)])) == 1

    def test_the_buffer_keeps_one_just_above_the_planned_altitude(self):
        """A pilot holds 6,500 to within a hundred feet, not to the foot, and
        may start a descent early."""
        assert len(keep([notam(lower_ft=7000.0, upper_ft=9000.0)])) == 1
        assert keep([notam(lower_ft=9000.0, upper_ft=12000.0)]) == ()

    def test_an_unstated_band_is_open_at_both_ends(self):
        kept = keep([notam(lower_ft=None, upper_ft=None)])
        assert len(kept) == 1
        assert any("could not be ruled out" in reason for reason in kept[0].reasons)

    def test_a_surface_notam_reaches_a_climb_that_starts_at_the_surface(self):
        """The leg's band is the band it covers, not its cruise altitude."""
        climb = leg(lower_ft=200.0, upper_ft=6500.0)
        assert len(keep([notam(lower_ft=0.0, upper_ft=500.0)], [climb])) == 1


class TestTheTimeWindow:
    def test_one_that_ended_before_the_flight_is_dropped(self):
        assert keep(
            [notam(effective_start=NOON - timedelta(days=1),
                   effective_end=NOON - timedelta(hours=1))]
        ) == ()

    def test_one_that_starts_after_the_flight_is_dropped(self):
        assert keep(
            [notam(effective_start=ONE + timedelta(hours=1),
                   effective_end=ONE + timedelta(days=1))]
        ) == ()

    def test_one_overlapping_the_flight_at_all_is_kept(self):
        assert len(keep([notam(effective_start=ONE - timedelta(minutes=5),
                               effective_end=ONE + timedelta(days=2))])) == 1

    def test_the_leg_window_is_finer_than_the_flight_window(self):
        """A NOTAM active for one hour is only relevant if the aeroplane is
        over *that* stretch during *that* hour.

        The first leg is flown at noon and the second at two; a restriction
        over the second leg from noon to one is about neither.
        """
        second = inverse(EAST, LatLon(37.0, -120.0))
        route = [
            leg(start=NOON, end=ONE),
            leg(span=second, start=NOON + timedelta(hours=2),
                end=NOON + timedelta(hours=3)),
        ]
        over_second_leg = notam(
            position=second.point_at_fraction(0.5),
            effective_start=NOON,
            effective_end=ONE,
        )
        assert nt.relevant([over_second_leg], route) == ()

    def test_and_keeps_it_when_the_aeroplane_is_there_in_time(self):
        second = inverse(EAST, LatLon(37.0, -120.0))
        route = [
            leg(start=NOON, end=ONE),
            leg(span=second, start=NOON + timedelta(hours=2),
                end=NOON + timedelta(hours=3)),
        ]
        over_second_leg = notam(
            position=second.point_at_fraction(0.5),
            effective_start=NOON + timedelta(hours=2, minutes=30),
            effective_end=NOON + timedelta(hours=4),
        )
        kept = nt.relevant([over_second_leg], route)
        assert len(kept) == 1
        assert kept[0].nearest_leg == 1

    def test_the_flight_window_prunes_before_any_geometry(self):
        """The coarse filter: next Tuesday's closure never reaches the map."""
        next_week = notam(
            effective_start=NOON + timedelta(days=7),
            effective_end=NOON + timedelta(days=8),
        )
        assert keep([next_week], start=NOON, end=ONE) == ()

    def test_a_notam_with_no_times_is_kept(self):
        kept = keep([notam(effective_start=None, effective_end=None)])
        assert len(kept) == 1
        assert any("no times" in reason for reason in kept[0].reasons)

    def test_a_permanent_notam_is_kept_and_labelled(self):
        kept = keep([notam(effective_end=None, permanent=True)])
        assert len(kept) == 1
        assert "permanent" in kept[0].reasons

    def test_an_estimated_end_is_labelled_because_it_is_not_a_commitment(self):
        kept = keep([notam(estimated_end=True)])
        assert "end time is an estimate" in kept[0].reasons


class TestAllFourAtOnce:
    def test_the_tests_must_be_met_by_the_same_leg(self):
        """Near the first leg, at the altitude of the second, at the time of
        neither -- that is about no part of this flight.

        Testing the three separately against the whole route would keep it.
        """
        second = inverse(EAST, LatLon(37.0, -120.0))
        route = [
            leg(lower_ft=1000.0, upper_ft=1000.0, start=NOON, end=ONE),
            leg(span=second, lower_ft=9500.0, upper_ft=9500.0,
                start=NOON + timedelta(hours=2), end=NOON + timedelta(hours=3)),
        ]
        awkward = notam(
            position=SPAN.point_at_fraction(0.5),  # on the first leg
            lower_ft=9000.0, upper_ft=11000.0,     # at the second leg's altitude
        )
        assert nt.relevant([awkward], route) == ()


class TestPriority:
    @pytest.mark.parametrize(
        "text",
        [
            "RWY 12/30 CLSD",
            "AD CLSD DUE TO SNOW",
            "TFR IN EFFECT FOR VIP MOVEMENT",
            "SQL VOR/DME OUT OF SERVICE",
            "GPS UNRELIABLE, GNSS UNUSABLE WI 30NM",
        ],
    )
    def test_the_things_that_stop_a_flight_are_critical(self, text):
        assert notam(text=text).priority is nt.Priority.CRITICAL

    @pytest.mark.parametrize(
        "text",
        ["CRANE 250FT AGL 2NM SW OF FIELD", "MOWING IN PROGRESS ADJ TWY B",
         "UAS OPERATIONS WI 1NM"],
    )
    def test_the_things_that_do_not_are_information(self, text):
        assert notam(text=text).priority is nt.Priority.INFORMATION

    def test_anything_unrecognised_is_operational_rather_than_buried(self):
        """The safe default: an unclassified NOTAM shown among the ordinary
        ones is read, and one filed under 'information' is not."""
        assert notam(text="APRON MARKINGS REPAINTED").priority is nt.Priority.OPERATIONAL

    def test_the_briefing_is_ordered_worst_first_then_nearest(self):
        entries = keep([
            notam(key="a", text="MOWING ADJ TWY B"),
            notam(key="b", text="RWY 12/30 CLSD", position=off_track(15.0)),
            notam(key="c", text="TWY A CLSD"),
            notam(key="d", text="RWY 06/24 CLSD"),
        ])
        assert [entry.notam.key for entry in entries] == ["d", "b", "c", "a"]
