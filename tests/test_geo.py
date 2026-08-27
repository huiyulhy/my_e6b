"""Geodesy and wind triangle tests.

The engine uses a spherical earth. What is checked here is that the spherical
solution is self-consistent -- `direct` and `inverse` are exact inverses, which
is what lets the navlog splice top of climb into a leg without the distances
drifting -- and that distances and bearings match independent hand-computed
references on legs at the scale a light aircraft flies.
"""

import math

import pytest

from engine import geo

# Reference airports, coordinates from the FAA Chart Supplement.
KSQL = geo.LatLon(37.5119, -122.2495)  # San Carlos, CA
KMRY = geo.LatLon(36.5870, -121.8429)  # Monterey, CA
KPAO = geo.LatLon(37.4611, -122.1150)  # Palo Alto, CA
KJFK = geo.LatLon(40.6398, -73.7789)  # New York
EGLL = geo.LatLon(51.4775, -0.4614)  # London Heathrow


class TestSphericalInverse:
    def test_known_transatlantic_distance(self):
        """JFK to Heathrow is about 3000 nm."""
        result = geo.inverse(KJFK, EGLL)
        assert result.distance_nm == pytest.approx(3000, rel=0.01)

    def test_short_local_leg(self):
        """San Carlos to Palo Alto, about 7 nm down the peninsula.

        Bounds here come from an independent flat-earth check
        (dlat/dlon in minutes, corrected for latitude), which gives 7.09 nm
        against an ellipsoidal 7.11 -- the difference being the ellipsoid.
        """
        result = geo.inverse(KSQL, KPAO)
        assert result.distance_nm == pytest.approx(7.1, abs=0.1)
        assert result.true_course_deg == pytest.approx(115.4, abs=0.5)

    def test_bay_area_leg_bearing_is_southeast(self):
        """San Carlos to Monterey: flat-earth check gives 58.8 nm at 160.7."""
        result = geo.inverse(KSQL, KMRY)
        assert result.distance_nm == pytest.approx(58.8, abs=0.3)
        assert 150.0 < result.true_course_deg < 170.0

    def test_identical_points(self):
        result = geo.inverse(KSQL, KSQL)
        assert result.distance_nm == 0.0

    def test_equatorial_line(self):
        """One degree of longitude at the equator is 60 nm."""
        result = geo.inverse(geo.LatLon(0.0, 0.0), geo.LatLon(0.0, 1.0))
        assert result.distance_nm == pytest.approx(60.1, rel=0.005)
        assert result.true_course_deg == pytest.approx(90.0)

    def test_meridian_arc_is_the_same_everywhere(self):
        """A degree of latitude is about 60 nm, and on a sphere it is uniform.

        This is the spherical model's defining limitation, asserted rather
        than hidden: the real earth is flattened, so a degree of latitude runs
        59.71 nm at the equator and 60.16 nm at 60 N. The sphere gives 60.04
        for both. That gap is no longer measured anywhere, since the
        ellipsoidal reference was removed; the point here is only that the
        model is internally consistent.

        An earlier version of this test asserted `polar > equatorial` to check
        for flattening. Under the sphere it kept passing -- but only by
        floating point accident, on a difference of 8e-14 nm caused by
        last-bit rounding in `radians(61) - radians(60)`. A test that passes
        for a reason unrelated to what it claims is worse than no test.
        """
        equatorial = geo.inverse(geo.LatLon(0.0, 0.0), geo.LatLon(1.0, 0.0))
        polar = geo.inverse(geo.LatLon(60.0, 0.0), geo.LatLon(61.0, 0.0))
        assert equatorial.distance_nm == pytest.approx(60.04, abs=0.01)
        assert polar.distance_nm == pytest.approx(equatorial.distance_nm, abs=1e-9)

    def test_due_north_bearing(self):
        result = geo.inverse(geo.LatLon(37.0, -122.0), geo.LatLon(38.0, -122.0))
        assert result.true_course_deg == pytest.approx(0.0, abs=1e-6)

    def test_crossing_the_antimeridian(self):
        result = geo.inverse(geo.LatLon(0.0, 179.5), geo.LatLon(0.0, -179.5))
        assert result.distance_nm == pytest.approx(60.1, rel=0.005)


class TestSphericalDirect:
    @pytest.mark.parametrize("bearing", [0.0, 45.0, 90.0, 137.0, 225.0, 359.0])
    @pytest.mark.parametrize("distance", [0.5, 10.0, 100.0, 1000.0])
    def test_round_trips_with_inverse(self, bearing, distance):
        """Fly out on a bearing, then measure back. Everything must agree.

        The direct and inverse methods share no code, so agreement to
        millimetres is a real check rather than a tautology.
        """
        start = geo.LatLon(37.5, -122.0)
        end = geo.direct(start, bearing, distance)
        back = geo.inverse(start, end)
        assert back.distance_nm == pytest.approx(distance, abs=1e-6)
        assert back.true_course_deg == pytest.approx(bearing % 360.0, abs=1e-6)

    def test_zero_distance(self):
        assert geo.direct(KSQL, 90.0, 0.0) == KSQL

    def test_east_along_the_equator(self):
        end = geo.direct(geo.LatLon(0.0, 0.0), 90.0, 60.1)
        assert end.lat == pytest.approx(0.0, abs=1e-9)
        assert end.lon == pytest.approx(1.0, rel=0.005)

    def test_longitude_stays_normalised_across_the_antimeridian(self):
        end = geo.direct(geo.LatLon(0.0, 179.0), 90.0, 200.0)
        assert -180.0 <= end.lon <= 180.0
        assert end.lon < 0.0


class TestCrossAndAlongTrack:
    def test_point_on_track_has_no_offset(self):
        leg = geo.inverse(KSQL, KMRY)
        assert leg.cross_track_nm(leg.point_at_fraction(0.5)) == pytest.approx(
            0.0, abs=0.01
        )

    def test_sign_indicates_which_side(self):
        """Due east of a northbound track is to the right, so positive."""
        leg = geo.inverse(geo.LatLon(37.0, -122.0), geo.LatLon(38.0, -122.0))
        assert leg.cross_track_nm(geo.LatLon(37.5, -121.8)) > 0
        assert leg.cross_track_nm(geo.LatLon(37.5, -122.2)) < 0

    def test_offset_magnitude(self):
        """A tenth of a degree of longitude at 37 N is about 4.8 nm."""
        leg = geo.inverse(geo.LatLon(37.0, -122.0), geo.LatLon(38.0, -122.0))
        assert abs(leg.cross_track_nm(geo.LatLon(37.5, -121.9))) == pytest.approx(
            4.8, rel=0.05
        )

    def test_along_track_at_the_ends(self):
        leg = geo.inverse(KSQL, KMRY)
        assert leg.along_track_nm(KSQL) == pytest.approx(0.0, abs=0.01)
        assert leg.along_track_nm(KMRY) == pytest.approx(leg.distance_nm, abs=0.1)

    def test_along_track_is_negative_before_the_start(self):
        """A point behind the departure end must report as behind it."""
        leg = geo.inverse(KSQL, KMRY)
        assert leg.along_track_nm(leg.point_at_nm(-10.0)) < 0

    def test_along_track_exceeds_the_leg_past_the_end(self):
        leg = geo.inverse(KSQL, KMRY)
        beyond = leg.point_at_nm(leg.distance_nm + 10.0)
        assert leg.along_track_nm(beyond) > leg.distance_nm


class TestPointsAlongTheSegment:
    def test_the_segment_carries_its_endpoints(self):
        leg = geo.inverse(KSQL, KMRY)
        assert leg.start == KSQL
        assert leg.end == KMRY

    def test_fraction_endpoints(self):
        leg = geo.inverse(KSQL, KMRY)
        start = leg.point_at_fraction(0.0)
        assert start.lat == pytest.approx(KSQL.lat, abs=1e-9)
        assert start.lon == pytest.approx(KSQL.lon, abs=1e-9)

    def test_midpoint_splits_the_distance(self):
        leg = geo.inverse(KSQL, KMRY)
        mid = leg.point_at_fraction(0.5)
        half = leg.distance_nm / 2
        assert geo.inverse(KSQL, mid).distance_nm == pytest.approx(half, abs=0.01)
        assert geo.inverse(mid, KMRY).distance_nm == pytest.approx(half, abs=0.01)

    def test_point_at_nm_matches_the_end(self):
        """Flying the full length must land on the destination."""
        leg = geo.inverse(KSQL, KMRY)
        end = leg.point_at_nm(leg.distance_nm)
        assert end.lat == pytest.approx(KMRY.lat, abs=1e-9)
        assert end.lon == pytest.approx(KMRY.lon, abs=1e-9)


class TestWindTriangle:
    def test_no_wind(self):
        result = geo.solve_wind_triangle(90.0, 110.0, 0.0, 0.0)
        assert result.true_heading_deg == pytest.approx(90.0)
        assert result.wind_correction_angle_deg == pytest.approx(0.0)
        assert result.ground_speed_kt == pytest.approx(110.0)

    def test_direct_headwind(self):
        """Wind from dead ahead: no crab, ground speed loses the full amount."""
        result = geo.solve_wind_triangle(90.0, 110.0, 90.0, 20.0)
        assert result.wind_correction_angle_deg == pytest.approx(0.0)
        assert result.ground_speed_kt == pytest.approx(90.0)
        assert result.headwind_kt == pytest.approx(20.0)

    def test_direct_tailwind(self):
        result = geo.solve_wind_triangle(90.0, 110.0, 270.0, 20.0)
        assert result.wind_correction_angle_deg == pytest.approx(0.0)
        assert result.ground_speed_kt == pytest.approx(130.0)
        assert result.headwind_kt == pytest.approx(-20.0)

    def test_direct_crosswind_from_the_right(self):
        """Crab into the wind, so heading goes right of course."""
        result = geo.solve_wind_triangle(0.0, 100.0, 90.0, 20.0)
        assert result.wind_correction_angle_deg > 0
        assert result.true_heading_deg == pytest.approx(11.54, abs=0.1)
        assert result.crosswind_kt == pytest.approx(20.0)
        # Pure crosswind still costs a little ground speed, via the crab.
        assert result.ground_speed_kt == pytest.approx(97.98, abs=0.1)

    def test_crosswind_from_the_left(self):
        result = geo.solve_wind_triangle(0.0, 100.0, 270.0, 20.0)
        assert result.wind_correction_angle_deg < 0
        assert result.true_heading_deg == pytest.approx(348.46, abs=0.1)

    def test_hand_worked_e6b_problem(self):
        """A worked example: course 090, TAS 110, wind from 140 at 25.

        Wind 50 deg off the nose from the right. Crosswind is
        25*sin(50) = 19.15 kt, so the crab is asin(19.15/110) = 10.02 deg
        right. Headwind is 25*cos(50) = 16.07 kt, and ground speed is
        110*cos(10.02) - 16.07 = 92.2 kt.
        """
        result = geo.solve_wind_triangle(90.0, 110.0, 140.0, 25.0)
        assert result.crosswind_kt == pytest.approx(19.15, abs=0.05)
        assert result.headwind_kt == pytest.approx(16.07, abs=0.05)
        assert result.wind_correction_angle_deg == pytest.approx(10.02, abs=0.05)
        assert result.true_heading_deg == pytest.approx(100.02, abs=0.05)
        assert result.ground_speed_kt == pytest.approx(92.23, abs=0.1)

    def test_heading_wraps_past_north(self):
        result = geo.solve_wind_triangle(355.0, 100.0, 85.0, 20.0)
        assert 0.0 <= result.true_heading_deg < 360.0
        assert result.true_heading_deg == pytest.approx(6.54, abs=0.1)

    def test_crosswind_stronger_than_airspeed_refuses(self):
        with pytest.raises(geo.WindTooStrong):
            geo.solve_wind_triangle(0.0, 20.0, 90.0, 60.0)

    def test_strong_headwind_is_allowed_even_if_ground_speed_is_negative(self):
        """Being blown backwards is a real answer; refusing would hide it."""
        result = geo.solve_wind_triangle(0.0, 60.0, 0.0, 80.0)
        assert result.ground_speed_kt < 0

    def test_zero_airspeed_rejected(self):
        with pytest.raises(ValueError):
            geo.solve_wind_triangle(90.0, 0.0, 90.0, 10.0)


class TestBearingHelpers:
    def test_normalise(self):
        assert geo.normalise_bearing(370.0) == pytest.approx(10.0)
        assert geo.normalise_bearing(-10.0) == pytest.approx(350.0)

    def test_difference_takes_the_short_way(self):
        assert geo.bearing_difference(350.0, 10.0) == pytest.approx(20.0)
        assert geo.bearing_difference(10.0, 350.0) == pytest.approx(-20.0)
        assert geo.bearing_difference(90.0, 100.0) == pytest.approx(10.0)

    def test_reversal_reports_a_half_turn(self):
        """Direction is arbitrary for an exact reversal; magnitude is not."""
        assert abs(geo.bearing_difference(0.0, 180.0)) == pytest.approx(180.0)


class TestLatLonValidation:
    def test_rejects_impossible_latitude(self):
        with pytest.raises(ValueError):
            geo.LatLon(91.0, 0.0)

    def test_rejects_impossible_longitude(self):
        with pytest.raises(ValueError):
            geo.LatLon(0.0, 181.0)

    def test_accepts_the_extremes(self):
        assert geo.LatLon(90.0, 180.0).lat == 90.0
        assert not math.isnan(geo.LatLon(-90.0, -180.0).lon)


class TestPolarRejection:
    """Near the poles the model refuses rather than returning a plausible number.

    Meridians converge so sharply there that a single "true course" stops
    describing the path, and the sphere's disagreement with the real earth is
    at its worst. Nothing this planner is for happens above 80 degrees.
    """

    POLAR = geo.LatLon(85.0, 10.0)
    TEMPERATE = geo.LatLon(37.5, -122.0)

    def test_limit_is_where_it_claims_to_be(self):
        assert geo.MAX_LATITUDE_DEG == 80.0

    def test_start_point_rejected(self):
        with pytest.raises(geo.PolarRegionUnsupported, match="start point"):
            geo.inverse(self.POLAR, self.TEMPERATE)

    def test_end_point_rejected(self):
        with pytest.raises(geo.PolarRegionUnsupported, match="end point"):
            geo.inverse(self.TEMPERATE, self.POLAR)

    def test_southern_hemisphere_rejected_too(self):
        with pytest.raises(geo.PolarRegionUnsupported):
            geo.inverse(geo.LatLon(-85.0, 0.0), self.TEMPERATE)

    def test_direct_rejects_a_polar_start(self):
        with pytest.raises(geo.PolarRegionUnsupported):
            geo.direct(self.POLAR, 90.0, 10.0)

    def test_direct_rejects_flying_into_the_polar_region(self):
        """Refused on arrival, not just departure -- the result is unusable either way."""
        near_limit = geo.LatLon(79.5, 0.0)
        with pytest.raises(geo.PolarRegionUnsupported, match="resulting point"):
            geo.direct(near_limit, 0.0, 120.0)

    def test_just_inside_the_limit_still_works(self):
        result = geo.inverse(geo.LatLon(79.9, 0.0), geo.LatLon(79.8, 1.0))
        assert result.distance_nm > 0

    def test_the_refusal_explains_itself(self):
        with pytest.raises(geo.PolarRegionUnsupported, match="80 degrees"):
            geo.inverse(self.POLAR, self.TEMPERATE)

    def test_a_polar_route_is_refused_by_the_navlog_too(self):
        """The refusal must reach the caller, not be swallowed en route."""
        from engine import navlog as nl

        route = [
            nl.Waypoint("NORTH", geo.LatLon(85.0, 10.0), "airport", elevation_ft=100),
            nl.Waypoint("ALSO", geo.LatLon(84.0, 20.0), "airport", elevation_ft=100),
        ]
        with pytest.raises(geo.PolarRegionUnsupported):
            nl.build_navlog(route, 6500, planning_mode="auto")
