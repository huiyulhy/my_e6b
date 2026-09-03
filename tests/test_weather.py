"""Decoding METAR, TAF and model payloads, and choosing between them.

Every test here reads a committed fixture captured from the live APIs, so the
suite stays offline and deterministic -- the fetchers in `server/` are what
touch the network, and `engine/weather.py` deliberately takes decoded payloads
so that it can be tested without mocking anything. That matches the rest of
this suite, which mocks nothing anywhere.

Times are derived from the fixtures themselves rather than from the clock. A
TAF fixture is a snapshot of a period that is now in the past, so a test
written against "two hours from now" would pass on the day it was recorded and
fail forever after.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from engine import weather as wx

FIXTURES = Path(__file__).parent / "fixtures" / "wx"


def load(name: str):
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


METAR_KSQL = load("metar_ksql.json")
TAF_KSFO = load("taf_ksfo.json")
TAF_KORD = load("taf_kord_tempo.json")
HISTORY = load("metar_history_bayarea.json")
MODEL_KSQL = load("openmeteo_ksql.json")


def block_time(payload, index: int) -> datetime:
    """A moment inside one TAF forecast block, from the fixture's own epochs."""
    block = payload[0]["fcsts"][index]
    start = datetime.fromtimestamp(block["timeFrom"], tz=UTC)
    end = datetime.fromtimestamp(block["timeTo"], tz=UTC)
    return start + (end - start) / 2


class TestUnits:
    """The altimeter unit is the one mistake that would be unsafe in silence."""

    def test_awc_hectopascals_become_inches(self):
        # The KSQL fixture reports altim 1012.6 and its rawOb says A2990.
        assert wx.hpa_to_inhg(1012.6) == pytest.approx(29.90, abs=0.005)

    def test_the_raw_metar_text_agrees_with_the_decoded_field(self):
        record = METAR_KSQL[0]
        assert "A2990" in record["rawOb"]
        assert wx.hpa_to_inhg(record["altim"]) == pytest.approx(29.90, abs=0.005)

    def test_hectopascals_mistaken_for_inches_are_refused(self):
        # 1012.6 read as inHg would be a pressure altitude of nonsense.
        with pytest.raises(ValueError, match="check the units"):
            wx.hpa_to_inhg(1012.6 * wx.HPA_PER_INHG)


class TestMetar:
    def test_a_single_observation_decodes(self):
        observation = wx.parse_metar(METAR_KSQL)
        assert observation is not None
        assert observation.station == "KSQL"
        assert observation.oat_c == pytest.approx(23.0)
        assert observation.altimeter_inhg == pytest.approx(29.90, abs=0.005)
        assert observation.valid_time.tzinfo is not None

    def test_it_reports_field_conditions_the_go_no_go_can_use(self):
        assert wx.parse_metar(METAR_KSQL).has_field_conditions

    def test_every_decoded_field_is_marked_as_coming_from_the_metar(self):
        observation = wx.parse_metar(METAR_KSQL)
        assert set(observation.sources.values()) == {wx.Source.METAR}
        assert observation.sources["altimeter_inhg"] is wx.Source.METAR

    def test_a_report_with_no_wind_group_has_no_wind_rather_than_calm(self):
        # The KSQL fixture's rawOb carries no wind group at all. Reporting
        # that as 000/00 would be inventing an observation.
        assert "KT" not in METAR_KSQL[0]["rawOb"]
        assert wx.parse_metar(METAR_KSQL).wind is None

    def test_an_empty_payload_is_none_not_an_exception(self):
        assert wx.parse_metar([]) is None

    def test_visibility_of_ten_plus_reads_as_ten(self):
        assert wx.parse_metar(METAR_KSQL).visibility_sm == pytest.approx(10.0)


class TestVisibilityStrings:
    """AWC returns `visib` as a number on some reports and a string on others."""

    def test_the_fixture_really_does_mix_both_types(self):
        kinds = {type(record.get("visib")).__name__ for record in HISTORY}
        assert kinds == {"str", "int"}

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (10, 10.0),
            ("10+", 10.0),
            ("6+", 6.0),
            ("1 1/2", 1.5),
            ("1/2", 0.5),
            ("", None),
            (None, None),
            ("M1/4", None),
        ],
    )
    def test_visibility_forms(self, raw, expected):
        got = wx._visibility_sm(raw)
        assert got is None if expected is None else got == pytest.approx(expected)


class TestMetarHistory:
    def test_every_requested_station_is_grouped(self):
        grouped = wx.parse_metar_history(HISTORY)
        assert {"KPAO", "KLVK", "KSQL", "KMOD", "KSFO", "KSJC"} <= set(grouped)

    def test_observations_come_back_in_time_order(self):
        for observations in wx.parse_metar_history(HISTORY).values():
            times = [item.valid_time for item in observations]
            assert times == sorted(times)

    def test_a_part_time_station_simply_has_fewer_observations(self):
        # KPAO does not report around the clock. That is a gap in the trend,
        # not a reason to drop the station or to fill the hours in.
        grouped = wx.parse_metar_history(HISTORY)
        assert 0 < len(grouped["KPAO"]) < len(grouped["KSQL"])

    def test_every_altimeter_setting_lands_in_a_plausible_band(self):
        for observations in wx.parse_metar_history(HISTORY).values():
            for item in observations:
                if item.altimeter_inhg is not None:
                    assert 28.0 <= item.altimeter_inhg <= 31.5


class TestTaf:
    def test_a_block_covering_the_target_is_found(self):
        target = block_time(TAF_KSFO, 0)
        forecast = wx.parse_taf(TAF_KSFO, target)
        assert forecast is not None
        assert forecast.station == "KSFO"
        assert forecast.wind is not None
        assert forecast.wind.from_deg == pytest.approx(300.0)
        assert forecast.wind.speed_kt == pytest.approx(15.0)
        assert forecast.gust_kt == pytest.approx(25.0)

    def test_a_taf_carries_no_temperature_and_no_altimeter(self):
        # This is the whole reason the model tier exists for surface weather.
        forecast = wx.parse_taf(TAF_KSFO, block_time(TAF_KSFO, 0))
        assert forecast.oat_c is None
        assert forecast.altimeter_inhg is None
        assert not forecast.has_field_conditions

    def test_a_later_fm_group_overrides_the_earlier_one(self):
        first = wx.parse_taf(TAF_KSFO, block_time(TAF_KSFO, 0))
        second = wx.parse_taf(TAF_KSFO, block_time(TAF_KSFO, 1))
        assert first.wind.from_deg == pytest.approx(300.0)
        assert second.wind.from_deg == pytest.approx(280.0)

    def test_a_target_beyond_the_valid_period_is_refused(self):
        end = datetime.fromtimestamp(TAF_KSFO[0]["validTimeTo"], tz=UTC)
        assert wx.parse_taf(TAF_KSFO, end + timedelta(hours=6)) is None

    def test_a_target_before_the_taf_was_issued_is_refused(self):
        start = datetime.fromtimestamp(TAF_KSFO[0]["validTimeFrom"], tz=UTC)
        assert wx.parse_taf(TAF_KSFO, start - timedelta(hours=6)) is None


class TestTransientGroups:
    """TEMPO and PROB describe the worst half-hour, not the planning value."""

    def test_the_base_group_wins_where_a_tempo_overlaps_it(self):
        # KORD's TEMPO 02:00-04:00 sits inside the base group 01:00-06:00.
        tempo = TAF_KORD[0]["fcsts"][1]
        assert tempo["fcstChange"] == "TEMPO"
        target = block_time(TAF_KORD, 1)
        forecast = wx.parse_taf(TAF_KORD, target)
        base = TAF_KORD[0]["fcsts"][0]
        assert forecast.wind.from_deg == pytest.approx(float(base["wdir"]))
        assert forecast.wind.speed_kt == pytest.approx(float(base["wspd"]))

    def test_but_the_tempo_is_reported_rather_than_hidden(self):
        forecast = wx.parse_taf(TAF_KORD, block_time(TAF_KORD, 1))
        assert any("TEMPO" in note for note in forecast.notes)

    def test_a_prob_group_names_its_probability_in_the_note(self):
        prob = TAF_KORD[0]["fcsts"][3]
        assert prob["fcstChange"] == "PROB" and prob["probability"] == 30
        forecast = wx.parse_taf(TAF_KORD, block_time(TAF_KORD, 3))
        assert any("30%" in note for note in forecast.notes)


class TestModelSurface:
    def test_the_hour_nearest_the_target_is_taken(self):
        first_hour = datetime.fromisoformat(MODEL_KSQL["hourly"]["time"][3]).replace(
            tzinfo=UTC
        )
        surface = wx.parse_model_surface(MODEL_KSQL, first_hour, station="KSQL")
        assert surface is not None
        assert surface.valid_time == first_hour
        assert surface.oat_c == pytest.approx(MODEL_KSQL["hourly"]["temperature_2m"][3])

    def test_the_altimeter_comes_from_sea_level_pressure(self):
        hour = datetime.fromisoformat(MODEL_KSQL["hourly"]["time"][0]).replace(tzinfo=UTC)
        surface = wx.parse_model_surface(MODEL_KSQL, hour, station="KSQL")
        expected = wx.hpa_to_inhg(MODEL_KSQL["hourly"]["pressure_msl"][0])
        assert surface.altimeter_inhg == pytest.approx(expected)

    def test_the_model_can_supply_field_conditions_on_its_own(self):
        hour = datetime.fromisoformat(MODEL_KSQL["hourly"]["time"][0]).replace(tzinfo=UTC)
        assert wx.parse_model_surface(MODEL_KSQL, hour, station="KSQL").has_field_conditions

    def test_a_target_outside_the_returned_window_is_refused(self):
        # Otherwise a request for next week silently snaps to the last hour.
        last = datetime.fromisoformat(MODEL_KSQL["hourly"]["time"][-1]).replace(tzinfo=UTC)
        assert wx.parse_model_surface(MODEL_KSQL, last + timedelta(days=3)) is None


class TestResolveNow:
    def test_a_current_metar_wins_on_every_field_it_publishes(self):
        metar = wx.parse_metar(METAR_KSQL)
        model = wx.parse_model_surface(
            MODEL_KSQL,
            datetime.fromisoformat(MODEL_KSQL["hourly"]["time"][0]).replace(tzinfo=UTC),
            station="KSQL",
        )
        resolved = wx.resolve_surface(
            station="KSQL", target=metar.valid_time, now=metar.valid_time,
            metar=metar, model=model,
        )
        assert resolved.sources["oat_c"] is wx.Source.METAR
        assert resolved.sources["altimeter_inhg"] is wx.Source.METAR
        assert resolved.oat_c == pytest.approx(metar.oat_c)

    def test_the_model_still_fills_a_field_the_metar_lacks(self):
        # The KSQL observation has no wind group; the model always has one.
        metar = wx.parse_metar(METAR_KSQL)
        model = wx.parse_model_surface(
            MODEL_KSQL,
            datetime.fromisoformat(MODEL_KSQL["hourly"]["time"][0]).replace(tzinfo=UTC),
            station="KSQL",
        )
        resolved = wx.resolve_surface(
            station="KSQL", target=metar.valid_time, now=metar.valid_time,
            metar=metar, model=model,
        )
        assert metar.wind is None
        assert resolved.wind is not None
        assert resolved.sources["wind"] is wx.Source.MODEL


class TestGustBelongsToTheWind:
    """A gust is a property of one wind, not a field resolved on its own.

    The case from the field: KHAF reports 00000KT while the model forecasts
    9 kt of gust. Resolving the two fields independently produces "calm,
    gusting 9" -- a peak that appeared in neither source, which the go/no-go
    would then resolve onto a runway as crosswind.
    """

    def model(self):
        return wx.parse_model_surface(
            MODEL_KSQL,
            datetime.fromisoformat(MODEL_KSQL["hourly"]["time"][0]).replace(tzinfo=UTC),
            station="KSQL",
        )

    def observed(self, **groups):
        """The KSQL observation with wind groups written into it."""
        record = dict(METAR_KSQL[0])
        record.update(groups)
        return wx.parse_metar([record])

    def resolve(self, metar):
        return wx.resolve_surface(
            station="KSQL", target=metar.valid_time, now=metar.valid_time,
            metar=metar, model=self.model(),
        )

    def test_a_model_gust_is_not_hung_on_an_observed_wind(self):
        metar = self.observed(wdir=0, wspd=0)
        assert metar.wind is not None and metar.gust_kt is None
        assert self.model().gust_kt is not None

        resolved = self.resolve(metar)
        assert resolved.sources["wind"] is wx.Source.METAR
        assert resolved.gust_kt is None
        assert "gust_kt" not in resolved.sources

    def test_and_says_that_it_dropped_one(self):
        notes = " ".join(self.resolve(self.observed(wdir=0, wspd=0)).notes)
        assert "gust" in notes and "dropped" in notes

    def test_an_observed_gust_is_kept(self):
        resolved = self.resolve(self.observed(wdir=290, wspd=14, wgst=22))
        assert resolved.gust_kt == pytest.approx(22.0)
        assert resolved.sources["gust_kt"] is wx.Source.METAR

    def test_a_model_gust_is_kept_when_the_wind_is_the_model_too(self):
        # The unmodified KSQL observation has no wind group at all, so both
        # halves fall to the model and there is nothing being mixed.
        resolved = self.resolve(wx.parse_metar(METAR_KSQL))
        assert resolved.sources["wind"] is wx.Source.MODEL
        assert resolved.sources["gust_kt"] is wx.Source.MODEL
        assert resolved.gust_kt is not None

    def test_a_gust_with_no_wind_at_all_is_dropped(self):
        stripped = wx.SurfaceWeather(
            station="KSQL",
            valid_time=self.model().valid_time,
            gust_kt=25.0,
            sources={"gust_kt": wx.Source.MODEL},
        )
        resolved = wx.resolve_surface(
            station="KSQL",
            target=stripped.valid_time,
            now=stripped.valid_time,
            model=stripped,
        )
        assert resolved.gust_kt is None
        assert "no wind for this time" in " ".join(resolved.notes)

    def test_a_model_gust_is_not_hung_on_a_forecast_wind_either(self):
        # TAF block 1 forecasts a wind with no gust; the model has one.
        target = block_time(TAF_KSFO, 1)
        resolved = wx.resolve_surface(
            station="KSFO", target=target, now=target - timedelta(hours=4),
            taf=wx.parse_taf(TAF_KSFO, target), model=self.model(),
        )
        assert resolved.sources["wind"] is wx.Source.TAF
        assert resolved.gust_kt is None


class TestResolveTargetTime:
    """The case the whole surface tier exists for: a go/no-go hours ahead."""

    def build(self, **overrides):
        target = block_time(TAF_KSFO, 1)
        metar = wx.parse_metar(METAR_KSQL)
        taf = wx.parse_taf(TAF_KSFO, target)
        model = wx.parse_model_surface(
            MODEL_KSQL,
            datetime.fromisoformat(MODEL_KSQL["hourly"]["time"][0]).replace(tzinfo=UTC),
            station="KSFO",
        )
        # `now` is pinned well before the target so this is a forecast case.
        kwargs = {
            "station": "KSFO",
            "target": target,
            "now": target - timedelta(hours=4),
            "metar": metar,
            "taf": taf,
            "model": model,
        }
        kwargs.update(overrides)
        return wx.resolve_surface(**kwargs)

    def test_wind_comes_from_the_taf(self):
        resolved = self.build()
        assert resolved.sources["wind"] is wx.Source.TAF
        assert resolved.wind.from_deg == pytest.approx(280.0)

    def test_temperature_and_altimeter_come_from_the_model(self):
        resolved = self.build()
        assert resolved.sources["oat_c"] is wx.Source.MODEL
        assert resolved.sources["altimeter_inhg"] is wx.Source.MODEL

    def test_the_result_can_therefore_produce_a_density_altitude(self):
        assert self.build().has_field_conditions

    def test_a_stale_metar_does_not_supply_the_temperature(self):
        # This is the trap: carrying this morning's observation into an
        # afternoon go/no-go is worst exactly when it matters most.
        resolved = self.build()
        assert resolved.sources["oat_c"] is not wx.Source.METAR

    def test_with_no_taf_the_model_supplies_everything(self):
        resolved = self.build(taf=None)
        assert resolved.sources["wind"] is wx.Source.MODEL
        assert resolved.has_field_conditions

    def test_with_no_model_there_is_no_density_altitude_and_it_says_so(self):
        resolved = self.build(model=None)
        assert not resolved.has_field_conditions
        assert any("cannot be computed" in note for note in resolved.notes)

    def test_nothing_at_all_yields_a_report_of_nones_rather_than_an_error(self):
        resolved = self.build(metar=None, taf=None, model=None)
        assert resolved.wind is None and resolved.oat_c is None
        assert resolved.altimeter_inhg is None
        assert resolved.station == "KSFO"


class TestBorrowedTaf:
    """Most fields a light aircraft uses publish no TAF of their own."""

    def build(self):
        target = block_time(TAF_KSFO, 1)
        return wx.resolve_surface(
            station="KSQL",
            target=target,
            now=target - timedelta(hours=4),
            taf=wx.parse_taf(TAF_KSFO, target),
            taf_station="KSFO",
            taf_distance_nm=11.0,
        )

    def test_a_borrowed_forecast_is_marked_as_borrowed(self):
        assert self.build().sources["wind"] is wx.Source.NEAREST_TAF

    def test_and_names_the_station_it_came_from(self):
        notes = " ".join(self.build().notes)
        assert "KSFO" in notes and "11 nm" in notes

    def test_the_station_asked_for_is_the_one_reported(self):
        assert self.build().station == "KSQL"


class TestTimezones:
    def test_a_naive_target_is_read_as_utc(self):
        target = block_time(TAF_KSFO, 1)
        aware = wx.parse_taf(TAF_KSFO, target)
        naive = wx.parse_taf(TAF_KSFO, target.replace(tzinfo=None))
        assert naive is not None
        assert naive.wind.from_deg == pytest.approx(aware.wind.from_deg)

    def test_resolved_times_always_carry_a_timezone(self):
        resolved = wx.resolve_surface(station="KSQL", metar=wx.parse_metar(METAR_KSQL))
        assert resolved.valid_time.tzinfo is not None


class TestSky:
    """The cloud group, which the go/no-go reads as more than a height.

    A ceiling height on its own cannot answer "is this field VFR": an overcast
    has to clear the traffic pattern, a broken layer only has to clear the
    regulation, and a vertical visibility is not a ceiling to fly under at
    all. So the cover comes back with the height -- and so does whether the
    sky was looked at, because a clear sky and a source that does not observe
    cloud both arrive with no ceiling in them.
    """

    def observed(self, **groups):
        record = dict(METAR_KSQL[0])
        record.update(groups)
        return wx.parse_metar([record])

    def forecast(self, **groups):
        payload = json.loads(json.dumps(TAF_KSFO))
        payload[0]["fcsts"][0].update(groups)
        return wx.parse_taf(payload, block_time(payload, 0))

    def test_an_empty_cloud_list_is_a_reported_clear_sky(self):
        # The KSQL fixture reports `"clouds": []`: looked at, nothing there.
        observation = wx.parse_metar(METAR_KSQL)
        assert observation.sky_reported is True
        assert observation.ceiling_ft_agl is None
        assert observation.ceiling_cover is None

    def test_a_record_with_no_cloud_group_is_not_a_clear_sky(self):
        record = {k: v for k, v in METAR_KSQL[0].items() if k != "clouds"}
        assert wx.parse_metar([record]).sky_reported is False

    def test_the_cover_that_made_the_ceiling_comes_back_with_it(self):
        observation = self.observed(
            clouds=[{"cover": "SCT", "base": 1200}, {"cover": "OVC", "base": 2500}]
        )
        assert observation.ceiling_ft_agl == pytest.approx(2500.0)
        assert observation.ceiling_cover == "OVC"

    def test_the_lowest_layer_that_is_a_ceiling_wins_not_the_lowest_layer(self):
        observation = self.observed(
            clouds=[{"cover": "FEW", "base": 500}, {"cover": "BKN", "base": 3000}]
        )
        assert observation.ceiling_ft_agl == pytest.approx(3000.0)
        assert observation.ceiling_cover == "BKN"

    def test_an_obscuration_is_reported_as_one(self):
        observation = self.observed(clouds=[{"cover": "OVX", "base": 200}])
        assert observation.ceiling_ft_agl == pytest.approx(200.0)
        assert observation.ceiling_cover == "OVX"

    def test_an_obscuration_with_no_height_is_still_an_obscuration(self):
        """The sky is hidden whether or not anybody measured how far up."""
        observation = self.observed(clouds=[{"cover": "OVX", "base": None}])
        assert observation.ceiling_cover == "OVX"
        assert observation.ceiling_ft_agl is None
        assert observation.sky_reported is True

    def test_a_taf_vertical_visibility_is_the_ceiling(self):
        forecast = self.forecast(vertVis=300, clouds=[])
        assert forecast.ceiling_ft_agl == pytest.approx(300.0)
        assert forecast.ceiling_cover == "VV"

    def test_a_vertical_visibility_under_a_cloud_layer_wins(self):
        forecast = self.forecast(vertVis=200, clouds=[{"cover": "BKN", "base": 1500}])
        assert forecast.ceiling_cover == "VV"

    def test_the_sky_is_resolved_whole_rather_than_field_by_field(self):
        """A METAR height must not end up wearing a TAF's cover code.

        The two describe different moments. Mixing them would report a sky
        that neither source saw, and the go/no-go would then decide on it.
        """
        metar = self.observed(clouds=[{"cover": "BKN", "base": 4000}])
        taf = wx.SurfaceWeather(
            station="KSQL",
            valid_time=metar.valid_time,
            ceiling_ft_agl=300.0,
            ceiling_cover="OVC",
            sky_reported=True,
        )
        resolved = wx.resolve_surface(
            station="KSQL", target=metar.valid_time, now=metar.valid_time,
            metar=metar, taf=taf,
        )
        # The observation is current, so it wins -- and it wins entire.
        assert resolved.ceiling_ft_agl == pytest.approx(4000.0)
        assert resolved.ceiling_cover == "BKN"

    def test_a_source_that_never_looked_does_not_overwrite_one_that_did(self):
        """A model has no cloud group, so it cannot clear a forecast sky."""
        taf = wx.SurfaceWeather(
            station="KSQL",
            valid_time=datetime(2026, 6, 1, 12, tzinfo=UTC),
            ceiling_ft_agl=800.0,
            ceiling_cover="OVC",
            sky_reported=True,
        )
        model = wx.SurfaceWeather(
            station="KSQL", valid_time=taf.valid_time, oat_c=18.0
        )
        resolved = wx.resolve_surface(
            station="KSQL", target=taf.valid_time, now=taf.valid_time,
            taf=taf, model=model,
        )
        assert resolved.ceiling_cover == "OVC"
        assert resolved.sky_reported is True

    def test_a_model_only_answer_reports_no_sky_at_all(self):
        model = wx.SurfaceWeather(
            station="KSQL",
            valid_time=datetime(2026, 6, 1, 12, tzinfo=UTC),
            oat_c=18.0,
        )
        resolved = wx.resolve_surface(
            station="KSQL", target=model.valid_time, now=model.valid_time, model=model
        )
        assert resolved.sky_reported is False
        assert resolved.ceiling_ft_agl is None
