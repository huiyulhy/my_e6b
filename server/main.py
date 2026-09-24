"""FastAPI development server.

**This is a dev convenience and is never shipped.** It exists so the browser
UI can reach the engine over HTTP while developing on a Linux desktop. In the
packaged PWA the same `ui/` files call the same engine functions through
Pyodide, with no server involved.

That is why the API surface here is kept deliberately narrow and thin: every
endpoint is a direct translation of an engine call, with no logic of its own.
Anything that starts to look like a decision belongs in `engine/`, or the two
run modes will drift apart. See docs/ARCHITECTURE.md section 2.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import airports as apt
from engine import aloft as al
from engine import atmosphere as atm
from engine import chart_render as render
from engine import charts as ch
from engine import consistency as consistency_checks
from engine import csv_download as csv_export
from engine import currency as cur
from engine import magnetic as mag
from engine import navlog as nl
from engine import notam as nt
from engine import planwx as pw
from engine import preflight as pf
from engine import weather as wx
from engine import weight_balance as wb
from engine.geo import LatLon, inverse
from engine.magnetic import AtGeographicPole, OutsideModelValidity
from server import notams as notam_source
from server import wx_surface

# How long a forecast window one request may ask for. A light single's day is
# a few hours, the model publishes 48, and a bound keeps a mistyped `hours=`
# from asking for all of them at every waypoint.
MAX_FORECAST_HOURS = 12

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
DATA_DIR = ROOT / "data"

# Uvicorn configures its own loggers and leaves the root one alone, so a
# warning raised anywhere in `engine/` or `server/` falls through to Python's
# handler of last resort -- which prints the bare message, with no level on
# it. That is the difference between a deployment whose log can be searched
# for "WARNING" and one an operator has to read line by line. Only added when
# nothing else has configured logging, so a host that brought its own setup
# keeps it.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:  %(name)s - %(message)s",
    )

app = FastAPI(title="my_e6b", description="Offline VFR cross-country planner")


# --- request and response models -----------------------------------------


class WaypointIn(BaseModel):
    name: str
    lat: float
    lon: float
    kind: str = "waypoint"
    elevation_ft: float | None = None
    # An intermediate airport the flight lands at rather than overflies.
    is_landing: bool = False
    # Identifier to look runways up by, for the go/no-go check. Sent
    # separately from `name` because a waypoint may have been renamed.
    ident: str | None = None
    # Cross this waypoint at this altitude instead of the cruise altitude,
    # typically to stay clear of airspace. Null means no constraint, and in
    # user-driven mode lets the altitude be computed from performance.
    altitude_ft: float | None = None
    # What the leg *arriving* here does: climb | cruise | descent | automatic.
    # Required to be concrete in user-driven mode; "automatic" asks the planner.
    segment_type: str = "automatic"
    # True for a TOC/TOD the planner inserted. Sent back so a later request can
    # be told which points to discard before re-planning.
    generated: bool = False
    # Field weather at an airport, which sets the density altitude the
    # takeoff and landing distances are read at. Null falls back to the
    # route-wide altimeter setting and ISA deviation.
    altimeter_inhg: float | None = None
    oat_c: float | None = None
    # The surface wind on the field, true-referenced as the METAR reports it.
    # It picks the runway end and corrects the distances, and its crosswind is
    # checked against the maximum demonstrated. Null leaves the go/no-go on
    # the no-wind book figures and says so.
    wind_from_deg: float | None = None
    wind_speed_kt: float | None = None
    gust_kt: float | None = None
    # The rest of the field's report, for the go/no-go's VFR check. Sent
    # together or not at all: `sky_reported` is what separates a reported
    # clear sky from a source that does not observe cloud, and without it a
    # missing ceiling could not be told from a missing report.
    visibility_sm: float | None = None
    ceiling_ft_agl: float | None = None
    ceiling_cover: str = ""  # BKN | OVC | OVX | VV
    sky_reported: bool = False
    # True once a report was obtained for this field, whatever was in it. A
    # model-only answer carries no cloud and no visibility, and without this
    # it would be indistinguishable from never having fetched at all.
    weather_reported: bool = False


class AloftLevelIn(BaseModel):
    """One pressure level of a forecast column, as `/api/wx/aloft` gave it.

    Sent back rather than re-fetched, so a plan is still one offline call: the
    UI fetched the column, and the plan endpoint stays a pure function of what
    it is handed.

    Both keys travel, because they are different altitudes for the same level.
    The wind is read at the geopotential height -- what a pilot holding 6500 ft
    on a correct setting is actually at -- and the temperature at the level's
    own pressure altitude, which is the coordinate the POH charts and every
    other station's report have in common.
    """

    height_ft: float | None = None
    wind_from_deg: float | None = None
    wind_speed_kt: float | None = None
    pressure_altitude_ft: float | None = None
    isa_deviation_c: float | None = None


class ForecastHourIn(BaseModel):
    """One hour of a forecast column, as `/api/wx/aloft/series` gave it."""

    valid_time: datetime
    levels: list[AloftLevelIn] = []


class PointForecastIn(BaseModel):
    """Every hour fetched over one waypoint of the route.

    One of these per waypoint the pilot drew, in route order. A window rather
    than a single hour because `engine/planwx` re-reads it as the times move:
    a leg's weather is chosen from the forecast at the hour that leg is
    actually reached, and that hour is not known until the plan is built.
    """

    name: str = ""
    lat: float
    lon: float
    hours: list[ForecastHourIn] = []


class LegOverrideIn(BaseModel):
    """A manually entered value for one navlog row."""

    row: int  # index into the returned legs
    # The wind on this leg. Unlike the rest of these it is applied before the
    # profile is solved, so it moves the top of climb and the top of descent as
    # well as the row's own ground speed; see navlog.LegOverride.
    wind_from_deg: float | None = None
    wind_speed_kt: float | None = None
    tas_kt: float | None = None
    # Level rows at or above 3000 ft pressure altitude only; see
    # navlog.LegOverride. Anything else comes back as ok:false with a reason.
    cruise_rpm: float | None = None
    # The altitude this leg ends at. Carries forward to every row after it.
    altitude_ft: float | None = None
    # Temperature at this row's altitude, off an FD forecast. Unlike the rest
    # it describes the air rather than the row, so it also lapses into the
    # rows around it; see navlog.LegOverride.
    oat_c: float | None = None
    # The pressure altitude this row's air is read at, replacing what the
    # route-wide altimeter setting implies. Density altitude follows from it
    # and the row's temperature; so do the charts the row reads. Stops at this
    # row, and cannot move a climb's published time and fuel -- a climb row
    # carrying one comes back with a warning saying so.
    pressure_altitude_ft: float | None = None


class PlanRequest(BaseModel):
    waypoints: list[WaypointIn]
    # "manual" -- the default -- takes the profile from each waypoint's
    # segment_type; "auto" lets the planner place TOC/TOD against
    # cruise_altitude_ft instead.
    planning_mode: str = "manual"
    cruise_altitude_ft: float = 6500
    cruise_rpm: float = 2400
    weight_lb: float = 2550
    fuel_on_board_gal: float = 50
    altimeter_inhg: float = 29.92
    isa_deviation_c: float = 0.0
    night: bool = False
    flight_date: date | None = None
    overrides: list[LegOverrideIn] = Field(default_factory=list)
    # One forecast series per waypoint the pilot drew, in route order, from
    # "Get weather". Each leg is then costed at both of its ends and planned
    # in whichever costs more. Empty is the ordinary case and means every leg
    # reads the wind typed on its row, or calm -- which is what a plan with
    # nothing entered has always meant.
    forecasts: list[PointForecastIn] = Field(default_factory=list)
    # How far either side of track a NOTAM still counts as being on the route.
    notam_corridor_nm: float = nt.DEFAULT_CORRIDOR_NM
    # When the flight leaves, UTC. What turns a cumulative ETE into a clock
    # time, and so what decides which forecast hour each leg is read at. Null
    # plans every leg on the hour the forecast was fetched for.
    off_blocks: datetime | None = None
    # Fractions: 0.20 means "require 20% more than the book distance".
    runway_margin: float = pf.DEFAULT_RUNWAY_MARGIN
    fuel_margin: float = pf.DEFAULT_FUEL_MARGIN


class StationIn(BaseModel):
    """One row of the loading form. Pounds and inches aft of the datum."""

    name: str = ""
    weight_lb: float
    arm_in: float


class WeightBalanceRequest(BaseModel):
    stations: list[StationIn]


# --- airport lookup ------------------------------------------------------


def _airport_json(airport: apt.Airport) -> dict:
    return {
        "ident": airport.ident,
        "name": airport.name,
        "kind": airport.kind,
        "lat": airport.position.lat,
        "lon": airport.position.lon,
        "elevation_ft": airport.elevation_ft,
        "municipality": airport.municipality,
        "region": airport.region,
        "longest_runway_ft": airport.longest_runway_ft,
        "label": airport.label,
    }


def _vfr_waypoint_json(waypoint: apt.VfrWaypoint) -> dict:
    """Shaped like an airport so the UI can treat search results uniformly.

    `elevation_ft` is deliberately null rather than zero: a VFR waypoint has
    no elevation, and a route cannot depart from or land at one. The navlog
    refuses a route whose endpoints lack elevation, which is the correct
    behaviour and depends on this staying null.
    """
    return {
        "ident": waypoint.ident,
        "name": "VFR checkpoint",
        "kind": "vfr_waypoint",
        "lat": waypoint.position.lat,
        "lon": waypoint.position.lon,
        "elevation_ft": None,
        "municipality": None,
        "region": f"US-{waypoint.state}" if waypoint.state else None,
        "longest_runway_ft": None,
        "label": waypoint.label,
    }


@app.get("/api/vfr-waypoints/bbox")
def vfr_waypoints_in_view(
    south: float, west: float, north: float, east: float, limit: int = 400
) -> list[dict]:
    """VFR waypoints in the visible map area."""
    return [
        _vfr_waypoint_json(w)
        for w in apt.vfr_waypoints_in_bounding_box(
            south, west, north, east, limit=limit
        )
    ]


@app.get("/api/airports/search")
def search_airports(q: str, limit: int = 15) -> list[dict]:
    """Search airports and published VFR waypoints together.

    One box rather than two, because a pilot picking the next point on a route
    does not first decide which kind of thing it is. Airports rank above
    waypoints on an equal match, since they are what a route usually starts
    and ends with.
    """
    airports = [_airport_json(a) for a in apt.search(q, limit=limit)]
    waypoints = [
        _vfr_waypoint_json(w)
        for w in apt.search_vfr_waypoints(q, limit=max(2, limit // 3))
    ]
    return (airports + waypoints)[:limit]


@app.get("/api/airports/bbox")
def airports_in_view(
    south: float,
    west: float,
    north: float,
    east: float,
    limit: int = 1500,
    min_runway_ft: float | None = None,
) -> list[dict]:
    """Airports in the visible map area.

    `min_runway_ft` lets the UI thin the display when zoomed out, where
    fifteen thousand markers would be unreadable anyway.
    """
    return [
        _airport_json(a)
        for a in apt.in_bounding_box(
            south, west, north, east, limit=limit, min_runway_ft=min_runway_ft
        )
    ]


@app.get("/api/airports/{ident}")
def get_airport(ident: str) -> dict:
    airport = apt.find(ident)
    if airport is None:
        raise HTTPException(404, f"no airport with identifier {ident!r}")
    return _airport_json(airport)


# --- weather -------------------------------------------------------------
# The one endpoint here that reaches the outside world. It is also the one
# place the two run modes genuinely differ rather than merely being wired
# differently: under Pyodide there is no server and no network, so there is no
# live weather and the pilot types the altimeter setting and temperature in as
# they always have. Everything downstream already works that way, which is why
# this can be additive.


def _surface_json(report: wx.SurfaceWeather) -> dict:
    """A resolved observation, in the units the UI's boxes already use.

    `sources` travels with it. A density altitude computed from a model is a
    different thing from one computed from an observation, and the pilot is
    entitled to see which they have before deciding to fly.
    """
    return {
        "station": report.station,
        "valid_time": report.valid_time.isoformat(),
        "wind_from_deg": None if report.wind is None else round(report.wind.from_deg, 1),
        "wind_speed_kt": None if report.wind is None else round(report.wind.speed_kt, 1),
        "gust_kt": report.gust_kt,
        "visibility_sm": report.visibility_sm,
        "ceiling_ft_agl": report.ceiling_ft_agl,
        # What kind of ceiling, and whether the sky was reported at all. The
        # go/no-go needs both: an overcast has to clear the pattern, and a
        # report with no sky group in it cannot say the field is VFR.
        "ceiling_cover": report.ceiling_cover,
        "sky_reported": report.sky_reported,
        # Rounded to what a pilot actually sets and reads: hundredths on the
        # Kollsman window, whole degrees on the ATIS.
        "oat_c": None if report.oat_c is None else round(report.oat_c, 1),
        "altimeter_inhg": (
            None if report.altimeter_inhg is None else round(report.altimeter_inhg, 2)
        ),
        "has_field_conditions": report.has_field_conditions,
        "sources": {name: str(source) for name, source in report.sources.items()},
        "notes": list(report.notes),
    }


def _aloft_json(forecast: al.AloftForecast) -> dict:
    """A forecast column, level by level, plus what it interpolates to.

    Both keys are given per level -- the geopotential height the wind is read
    at and the pressure altitude the temperature is read at -- because they are
    different altitudes for the same level and a caller comparing this against
    a navlog row needs to know which one it is looking at.
    """
    return {
        "ok": True,
        "valid_time": forecast.valid_time.isoformat(),
        "terrain_elevation_ft": forecast.terrain_elevation_ft,
        "levels": [
            {
                "pressure_hpa": level.pressure_hpa,
                "height_ft": round(level.height_ft, 1),
                "pressure_altitude_ft": round(level.pressure_altitude_ft, 1),
                "wind_from_deg": None if level.wind is None else round(level.wind.from_deg, 1),
                "wind_speed_kt": None if level.wind is None else round(level.wind.speed_kt, 1),
                "oat_c": None if level.oat_c is None else round(level.oat_c, 1),
                "isa_deviation_c": (
                    None if level.isa_deviation_c is None
                    else round(level.isa_deviation_c, 1)
                ),
            }
            for level in forecast.levels
        ],
        "notes": list(forecast.notes),
    }


@app.get("/api/wx/aloft")
def winds_aloft(
    lat: float,
    lon: float,
    time: str | None = None,
    ceiling_ft: float = 14000.0,
) -> dict:
    """Winds and temperatures aloft over one point.

    By position rather than by identifier, unlike the surface endpoint: the
    wind at cruise belongs to the sky the leg crosses, and reading it off the
    departure airport is the FD product's mistake rather than a shortcut.
    """
    target: datetime | None = None
    if time:
        try:
            target = datetime.fromisoformat(time)
        except ValueError:
            raise HTTPException(400, f"could not read {time!r} as an ISO 8601 time")

    try:
        forecast = wx_surface.fetch_aloft(
            LatLon(lat, lon), target, ceiling_ft=ceiling_ft
        )
    except ValueError as exc:
        # LatLon refuses an impossible position; that is a bad request, not a
        # weather outage.
        raise HTTPException(400, str(exc))
    except wx.WeatherUnavailable as exc:
        return {"ok": False, "error": str(exc)}
    return _aloft_json(forecast)


@app.get("/api/wx/aloft/series")
def winds_aloft_series(
    lat: float,
    lon: float,
    time: str | None = None,
    hours: int = 1,
    ceiling_ft: float = 14000.0,
) -> dict:
    """Consecutive forecast hours over one point, in one request.

    A window rather than an hour, because the planner cannot know which hour
    it needs until it knows when the leg is reached -- and it cannot know that
    until it has planned it. `engine/planwx.solve` settles the two against
    each other, re-reading the window on every pass, so the window has to be
    in hand before the loop starts. It costs one upstream request either way:
    the model publishes the whole day and the cache is keyed by it.
    """
    if not 1 <= hours <= MAX_FORECAST_HOURS:
        raise HTTPException(
            400, f"hours must be between 1 and {MAX_FORECAST_HOURS}"
        )
    start: datetime | None = None
    if time:
        try:
            start = datetime.fromisoformat(time)
        except ValueError:
            raise HTTPException(400, f"could not read {time!r} as an ISO 8601 time")

    try:
        series = wx_surface.fetch_aloft_series(
            LatLon(lat, lon), start, hours=hours, ceiling_ft=ceiling_ft
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except wx.WeatherUnavailable as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "hours": [_aloft_json(forecast) for forecast in series]}


@app.get("/api/wx/surface")
def surface_weather(ident: str, time: str | None = None) -> dict:
    """Surface weather at one field, for now or for a target time.

    `time` is an ISO 8601 instant; naive values are read as UTC, as every
    source here publishes in Zulu.
    """
    target: datetime | None = None
    if time:
        try:
            target = datetime.fromisoformat(time)
        except ValueError:
            raise HTTPException(400, f"could not read {time!r} as an ISO 8601 time")

    try:
        report = wx_surface.fetch_surface(ident, target)
    except wx_surface.AirportUnknown as exc:
        raise HTTPException(404, str(exc))
    except wx.WeatherUnavailable as exc:
        # The established convention in this file: an engine-level refusal is
        # data the UI renders, not an HTTP error it has to catch.
        return {"ok": False, "error": str(exc)}

    return {"ok": True, **_surface_json(report)}


# --- planning ------------------------------------------------------------


def _waypoint_json(waypoint: nl.Waypoint, conditions: nl.Conditions) -> dict:
    """One resolved waypoint, in the shape the UI sends waypoints back in.

    Plus the air at it. The `field_*` keys are derived, never sent back: they
    are what the elevation and altimeter setting on this waypoint work out to,
    shown beside the boxes they came from so that typing a setting has a
    visible consequence instead of disappearing into the plan. Null where the
    waypoint has no elevation to read them at.
    """
    air = nl.field_weather(waypoint, conditions)
    return {
        "name": waypoint.name,
        "lat": waypoint.position.lat,
        "lon": waypoint.position.lon,
        "kind": waypoint.kind,
        "elevation_ft": waypoint.elevation_ft,
        "is_landing": waypoint.is_landing,
        "altitude_ft": waypoint.altitude_ft,
        "segment_type": waypoint.segment_type,
        "generated": waypoint.generated,
        "altimeter_inhg": waypoint.altimeter_inhg,
        "oat_c": waypoint.oat_c,
        "wind_from_deg": waypoint.wind_from_deg,
        "wind_speed_kt": waypoint.wind_speed_kt,
        "gust_kt": waypoint.gust_kt,
        # The setting and temperature actually used, which is the waypoint's
        # own where it has one and the route's where it does not.
        "field_altimeter_inhg": None if air is None else air.altimeter_inhg,
        "field_oat_c": None if air is None else air.oat_c,
        "field_pressure_altitude_ft": (
            None if air is None else air.pressure_altitude_ft
        ),
        "field_density_altitude_ft": (
            None if air is None else air.density_altitude_ft
        ),
    }


def _build_from_request(request: PlanRequest):
    """Turn a request into the engine objects a plan is built from.

    Shared by `/api/plan` and `/api/consistency` so the two can never disagree
    about what the plan being described actually is.
    """
    waypoints = [
        nl.Waypoint(
            name=w.name,
            position=LatLon(w.lat, w.lon),
            kind=w.kind,
            elevation_ft=w.elevation_ft,
            is_landing=w.is_landing,
            altitude_ft=w.altitude_ft,
            segment_type=w.segment_type,
            generated=w.generated,
            altimeter_inhg=w.altimeter_inhg,
            oat_c=w.oat_c,
            wind_from_deg=w.wind_from_deg,
            wind_speed_kt=w.wind_speed_kt,
            gust_kt=w.gust_kt,
            visibility_sm=w.visibility_sm,
            ceiling_ft_agl=w.ceiling_ft_agl,
            ceiling_cover=w.ceiling_cover,
            sky_reported=w.sky_reported,
            weather_reported=w.weather_reported,
            runways=_runways_for(w),
            # Looked up here rather than sent, for the same reason the runways
            # are: it is a fact about the field, not about the plan, and the
            # database already on disk is a better source than a round trip.
            pattern_altitude_agl_ft=_pattern_altitude_for(w),
        )
        for w in request.waypoints
    ]
    # No route-wide winds or temperatures aloft: an FD level is one number for
    # a quarter of a state, and the wind on the coast is not the wind over the
    # valley. Wind is entered per navlog row instead, where it belongs to the
    # leg it was forecast for, and temperature comes from the fields plus any
    # row the pilot typed one on. Calm here is what a row with no wind on it
    # means, not a claim about the day.
    conditions = nl.Conditions(
        altimeter_inhg=request.altimeter_inhg,
        isa_deviation_c=request.isa_deviation_c,
        flight_date=request.flight_date,
        night=request.night,
    )
    aircraft = nl.Aircraft(
        weight_lb=request.weight_lb,
        cruise_rpm=request.cruise_rpm,
        fuel_on_board_gal=request.fuel_on_board_gal,
    )
    overrides = {
        o.row: nl.LegOverride(
            wind_from_deg=o.wind_from_deg,
            wind_speed_kt=o.wind_speed_kt,
            tas_kt=o.tas_kt,
            cruise_rpm=o.cruise_rpm,
            altitude_ft=o.altitude_ft,
            oat_c=o.oat_c,
            pressure_altitude_ft=o.pressure_altitude_ft,
        )
        for o in request.overrides
    }
    margins = pf.Margins(runway=request.runway_margin, fuel=request.fuel_margin)
    # Not `build_navlog` directly: with a forecast on the route the weather
    # and the log are each an input to the other, and `planwx.solve` settles
    # them against each other. With no forecast it builds the log once and
    # returns, so a plan with nothing fetched costs exactly what it did.
    solved = pw.solve(
        waypoints,
        request.cruise_altitude_ft,
        aircraft,
        conditions,
        forecasts=tuple(_point_forecast(f) for f in request.forecasts),
        off_blocks=request.off_blocks,
        overrides=overrides,
        margins=margins,
        planning_mode=request.planning_mode,
    )
    return solved, aircraft, conditions


@app.post("/api/consistency")
def consistency(request: PlanRequest) -> dict:
    """Check a finished navlog for internal contradictions.

    Its own endpoint rather than part of `/api/plan`, which runs on every
    keystroke: this is a deliberate act by the pilot, and reads as one.
    """
    if len(request.waypoints) < 2:
        return {"ok": False, "error": "Add a departure and a destination."}
    try:
        solved, aircraft, conditions = _build_from_request(request)
        log = solved.navlog
    except (nl.RouteError, OutsideModelValidity) as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    # The log's own conditions, not the request's: the temperature profile is
    # folded in while the log is built, so this is the only object that knows
    # what the air was taken to be. Checking against the request's would grade
    # the plan on a day it was not planned for.
    report = consistency_checks.check_navlog_consistency(
        log, aircraft=aircraft, conditions=log.conditions or conditions
    )
    return {
        "ok": True,
        "is_consistent": report.is_consistent,
        "findings": [
            {
                "severity": f.severity,
                "code": f.code,
                "message": f.message,
                "row": f.row,
            }
            for f in report.findings
        ],
        "text": consistency_checks.format_report(report),
    }


@app.post("/api/navlog.csv")
def navlog_csv(request: PlanRequest, time_off: str | None = None) -> Response:
    """The same plan, as the columns of a paper navigation log.

    A download rather than JSON, so the browser saves a file the pilot can open
    in a spreadsheet. Everything about the format lives in
    `engine.csv_download`; this only builds the plan and attaches a filename.

    `time_off` is an optional `HH:MM` departure time, which fills the ETA
    column. Errors come back as JSON like every other endpoint's: a browser
    that asked for a file and got a 400 can still read the reason out of it.
    """
    if len(request.waypoints) < 2:
        return JSONResponse(
            {"ok": False, "error": "Add a departure and a destination."},
            status_code=400,
        )
    try:
        solved, _aircraft, _conditions = _build_from_request(request)
        log = solved.navlog
        text = csv_export.navlog_csv(log, time_off=time_off)
    except (nl.RouteError, OutsideModelValidity, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    return Response(
        content=text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{csv_export.csv_filename(log)}"'
            )
        },
    )


@app.post("/api/notams")
def notams(request: PlanRequest) -> dict:
    """The NOTAMs that are actually about this flight.

    Its own endpoint rather than part of `/api/plan`, on the same reasoning as
    the consistency check: `/api/plan` runs on every keystroke, and a NOTAM
    briefing is a deliberate act with a rate-limited external service behind
    it.

    The route is planned first, because the filter needs what only a plan
    knows -- the altitude over each stretch of ground and the clock time the
    aeroplane is there. Then SkyLink is asked about every aerodrome and centre
    along the route, and the four tests in `engine/notam` sort what comes back.
    """
    if len(request.waypoints) < 2:
        return {"ok": False, "error": "Add a departure and a destination."}
    if not notam_source.credentials_configured():
        return {
            "ok": False,
            "error": (
                f"NOTAMs need a SkyLink subscription on RapidAPI. Set "
                f"{notam_source.RAPIDAPI_KEY_ENV} before starting the server."
            ),
            "needs_credentials": True,
        }

    try:
        solved, _aircraft, _conditions = _build_from_request(request)
    except (nl.RouteError, OutsideModelValidity) as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    window = _route_window(solved.navlog, request.off_blocks)
    if not window:
        return {"ok": False, "error": "This route has no legs to search along."}

    try:
        found = notam_source.fetch_route(
            [leg.span.start for leg in window] + [window[-1].span.end],
            corridor_nm=request.notam_corridor_nm,
            # The route's own fields go to the front of the queue, so the
            # request cap can never cut the departure or the destination.
            priority=tuple(
                w.ident or w.name for w in request.waypoints if _is_airport(w)
            ),
        )
    except notam_source.NotamsUnavailable as exc:
        return {"ok": False, "error": str(exc)}

    kept = nt.relevant(
        found.notams,
        window,
        corridor_nm=request.notam_corridor_nm,
        start=window[0].start,
        end=window[-1].end,
    )
    return {
        "ok": True,
        # Both numbers, always. "4 NOTAMs" reads very differently once you
        # know it came from 137 -- and a filter that is throwing away almost
        # everything is one worth being able to see working.
        "returned": len(found.notams),
        "relevant": len(kept),
        "corridor_nm": request.notam_corridor_nm,
        # What SkyLink was asked about: the aerodromes and centres along the
        # route, so a pilot can see which fields the briefing covers.
        "designators": list(found.designators),
        # A partial search is worth showing, and worth labelling. An empty
        # NOTAM list is the most dangerous thing this endpoint can return.
        "complete": found.complete,
        "failed": list(found.failed),
        "window": {
            "start": None if window[0].start is None else window[0].start.isoformat(),
            "end": None if window[-1].end is None else window[-1].end.isoformat(),
        },
        "notams": [_notam_json(entry) for entry in kept],
    }


def _route_window(log: nl.Navlog, off_blocks: datetime | None) -> list[nt.RouteWindow]:
    """Where the flight goes, how high and when, one entry per navlog row.

    Per row rather than per drawn leg, because the altitude is the whole point
    of the vertical test: a climb out of a field covers the surface to 6,500
    ft and the cruise after it covers only 6,500. Merged into one leg they
    would together claim every altitude over the whole route.

    Rows that cover no ground -- the taxi, the pattern -- have no track for a
    corridor and are left out; the fields they happen at are the ends of the
    legs either side of them.
    """
    window: list[nt.RouteWindow] = []
    start = None if off_blocks is None else _utc(off_blocks)
    for leg in log.legs:
        if not leg.covers_ground:
            continue
        entry = leg.entry_altitude_ft
        exit_ = leg.exit_altitude_ft
        low = leg.altitude_ft if entry is None or exit_ is None else min(entry, exit_)
        high = leg.altitude_ft if entry is None or exit_ is None else max(entry, exit_)
        window.append(
            nt.RouteWindow(
                span=inverse(leg.from_position, leg.to_position),
                lower_ft=low,
                upper_ft=high,
                start=(
                    None
                    if start is None
                    else start
                    + timedelta(minutes=leg.cumulative_ete_min - leg.ete_min)
                ),
                end=(
                    None
                    if start is None
                    else start + timedelta(minutes=leg.cumulative_ete_min)
                ),
            )
        )
    return window


def _notam_json(entry: nt.Relevance) -> dict:
    one = entry.notam
    return {
        "key": one.key,
        "number": one.number,
        "location": one.location,
        "text": one.text,
        "priority": str(one.priority),
        "distance_nm": None if entry.distance_nm is None else round(entry.distance_nm, 1),
        "nearest_leg": entry.nearest_leg,
        "lat": None if one.position is None else round(one.position.lat, 4),
        "lon": None if one.position is None else round(one.position.lon, 4),
        "radius_nm": one.radius_nm,
        "lower_ft": one.lower_ft,
        "upper_ft": one.upper_ft,
        "effective_start": (
            None if one.effective_start is None else one.effective_start.isoformat()
        ),
        "effective_end": (
            None if one.effective_end is None else one.effective_end.isoformat()
        ),
        "permanent": one.permanent,
        "estimated_end": one.estimated_end,
        # Why it survived the filter, in the filter's own terms.
        "reasons": list(entry.reasons),
    }


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


@app.post("/api/plan")
def plan(request: PlanRequest) -> dict:
    """Build a navigation log for a route.

    Engine errors are returned as a structured payload rather than an HTTP
    error, so the UI can show "cruise altitude is below the destination"
    beside the form instead of a stack trace. Only genuinely unexpected
    failures become 500s.
    """
    if len(request.waypoints) < 2:
        return {"ok": False, "error": "Add a departure and a destination."}

    try:
        solved, _aircraft, conditions = _build_from_request(request)
        log = solved.navlog
    except (nl.RouteError, OutsideModelValidity) as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError as exc:
        # Includes OutsidePOHEnvelope and AboveModelCeiling -- a refusal to
        # invent numbers is a real answer, so it reads as one.
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "legs": [
            {
                "from": leg.from_name,
                "to": leg.to_name,
                "phase": leg.phase,
                # False for the taxi and traffic-pattern rows, which burn time
                # and fuel at one point and have no course or heading to show.
                "covers_ground": leg.covers_ground,
                "altitude_ft": leg.altitude_ft,
                "from_lat": leg.from_position.lat,
                "from_lon": leg.from_position.lon,
                "to_lat": leg.to_position.lat,
                "to_lon": leg.to_position.lon,
                "true_course_deg": leg.true_course_deg,
                "variation_deg": leg.variation_deg,
                "magnetic_course_deg": leg.magnetic_course_deg,
                "wind_correction_angle_deg": leg.wind_correction_angle_deg,
                "true_heading_deg": leg.true_heading_deg,
                "magnetic_heading_deg": leg.magnetic_heading_deg,
                "wind_from_deg": leg.wind_from_deg,
                "wind_speed_kt": leg.wind_speed_kt,
                "headwind_kt": leg.headwind_kt,
                "tas_kt": leg.tas_kt,
                "ground_speed_kt": leg.ground_speed_kt,
                "distance_nm": leg.distance_nm,
                "cumulative_distance_nm": leg.cumulative_distance_nm,
                "ete_min": leg.ete_min,
                "cumulative_ete_min": leg.cumulative_ete_min,
                "fuel_gal": leg.fuel_gal,
                "fuel_remaining_gal": leg.fuel_remaining_gal,
                "overridden": list(leg.overridden),
                # Whether this row's performance came off the published chart
                # or from beside it. The row marks itself so a pilot reading
                # the log can see where the totals stop being the book's.
                "extrapolated": leg.extrapolated,
                "off_chart": [
                    {
                        "what": entry.what,
                        "detail": entry.detail,
                        "conservative": entry.conservative,
                    }
                    for entry in leg.off_chart
                ],
                "cruise_rpm": leg.cruise_rpm,
                "flight_index": leg.flight_index,
                "segment_type": leg.segment_type,
                "entry_altitude_ft": leg.entry_altitude_ft,
                "exit_altitude_ft": leg.exit_altitude_ft,
                # The air this row was flown through, at its altitude. Every
                # performance figure on the row was read at this density
                # altitude.
                "oat_c": leg.oat_c,
                "pressure_altitude_ft": leg.pressure_altitude_ft,
                "density_altitude_ft": leg.density_altitude_ft,
                # "TOC" / "TOD" beside the name, never replacing it.
                "start_role": leg.start_role,
                "end_role": leg.end_role,
            }
            for leg in log.legs
        ],
        "checklist": _checklist_json(log.checklist),
        # The route the rows were built from. In automatic mode this has the
        # TOC/TOD the planner inserted, so the UI can show them in the waypoint
        # list and let the pilot take them over.
        "planning_mode": log.planning_mode,
        # Read off the log's own conditions, not the request's: the
        # temperature profile is folded in while the log is built, so this is
        # the only object that knows what the air was actually taken to be.
        "resolved_waypoints": [
            _waypoint_json(w, log.conditions or conditions)
            for w in log.resolved_waypoints
        ],
        "totals": {
            "distance_nm": log.total_distance_nm,
            "time_min": log.total_time_min,
            "fuel_gal": log.total_fuel_gal,
            "fuel_remaining_gal": log.fuel_remaining_gal,
            "reserve_required_gal": log.reserve_required_gal,
            "legal_on_fuel": log.is_legal_on_fuel,
            # The row from which the totals stop being the book's: fuel and
            # time are cumulative, so an off-chart reading taints everything
            # downstream of where it first appears. Null when every row came
            # straight off the chart.
            "first_extrapolated_leg": log.first_extrapolated_leg,
            "extrapolated_legs": list(log.extrapolated_legs),
        },
        # The second of the two lists: the weather each leg the pilot drew was
        # planned in, and which end of the leg it came from. Empty when
        # nothing was fetched.
        "weather": [_leg_weather_json(entry) for entry in solved.weather],
        "weather_passes": solved.passes,
        "weather_settled": solved.settled,
        "warnings": list(log.warnings) + list(solved.notes),
        "text": nl.format_navlog(log),
    }


def _leg_weather_json(entry: pw.LegWeather) -> dict:
    def wind(value):
        if value is None:
            return None
        return {
            "from_deg": round(value.from_deg, 1),
            "speed_kt": round(value.speed_kt, 1),
        }

    return {
        "leg": entry.leg,
        "from": entry.from_name,
        "to": entry.to_name,
        # "start" | "end" | "" -- which end of the leg the forecast came from.
        "chosen": entry.chosen,
        "valid_time": None if entry.valid_time is None else entry.valid_time.isoformat(),
        # What each end was worth over this leg. The comparison that decided
        # it, shown rather than asserted.
        "start_fuel_gal": _finite(entry.start_fuel_gal),
        "end_fuel_gal": _finite(entry.end_fuel_gal),
        "start_wind": wind(entry.start_wind),
        "end_wind": wind(entry.end_wind),
        "note": entry.note,
    }


def _finite(value: float | None) -> float | None:
    """JSON has no infinity, and an unflyable leg is reported as one anyway."""
    if value is None or value != value or value in (float("inf"), float("-inf")):
        return None
    return round(value, 2)


# What the UI can send as a waypoint's kind when it means "a field". The map
# popup stamps a bare "airport"; a search result carries the database's own
# `large_airport` / `medium_airport` / `small_airport`, which is the size of
# the place and not a different sort of thing. Matching only the bare word
# silently skipped the runway lookup for every airport added from the search
# box, and the go/no-go reported "no runway data on file" for fields whose
# runways are in the database it had just searched.
def _is_airport(waypoint: WaypointIn) -> bool:
    return waypoint.kind == "airport" or waypoint.kind.endswith("_airport")


def _runways_for(waypoint: WaypointIn) -> tuple[pf.Runway, ...]:
    """Runways for a waypoint, if it is an airport we know about."""
    ident = waypoint.ident or waypoint.name
    if not _is_airport(waypoint):
        return ()
    try:
        return tuple(apt.runways(ident))
    except apt.AirportDatabaseMissing:
        return ()


def _point_forecast(point: PointForecastIn) -> pw.PointForecast:
    """One waypoint's fetched window as the engine's own type."""
    return pw.PointForecast(
        name=point.name,
        position=LatLon(point.lat, point.lon),
        hours=tuple(
            pw.ForecastHour(
                valid_time=hour.valid_time,
                winds=_winds_aloft(hour),
                temperatures=tuple(_aloft_temperature_samples(hour)),
            )
            for hour in point.hours
        ),
    )


def _winds_aloft(hour: ForecastHourIn) -> nl.WindsAloft:
    """One hour's levels as a wind profile, keyed by geopotential height.

    Height rather than pressure altitude: it is where the aeroplane actually
    is. Levels with half a wind are dropped rather than half-read -- a
    direction with no speed would interpolate as a calm from that bearing,
    which is a claim about the air rather than the absence of one.
    """
    return nl.WindsAloft(
        tuple(
            (level.height_ft, nl.Wind(level.wind_from_deg, level.wind_speed_kt))
            for level in hour.levels
            if level.height_ft is not None
            and level.wind_from_deg is not None
            and level.wind_speed_kt is not None
        )
    )


def _aloft_temperature_samples(hour: ForecastHourIn) -> list[nl.TemperatureSample]:
    """The hour's temperatures, keyed by the pressure altitude they are at.

    No conversion, which is the point of the pressure-level frame: a field
    temperature has to be read through its station's altimeter setting before
    it can be compared with anything, and a constant-pressure surface is
    already in the coordinate they all share.
    """
    return [
        nl.TemperatureSample(level.pressure_altitude_ft, level.isa_deviation_c)
        for level in hour.levels
        if level.pressure_altitude_ft is not None and level.isa_deviation_c is not None
    ]


def _pattern_altitude_for(waypoint: WaypointIn) -> float | None:
    """The field's published pattern altitude in feet AGL, where there is one.

    `None` for anything that is not an airport we know, and for the great
    majority of airports that are: NASR only carries a pattern altitude where
    somebody filed one. The checklist takes the standard 1,000 ft in that case
    rather than reading the silence as a low pattern.
    """
    ident = waypoint.ident or waypoint.name
    if not _is_airport(waypoint):
        return None
    try:
        airport = apt.find(ident)
    except apt.AirportDatabaseMissing:
        return None
    return None if airport is None else airport.pattern_altitude_agl_ft


def _checklist_json(checklist: pf.GoNoGo | None) -> dict | None:
    if checklist is None:
        return None
    return {
        "is_go": checklist.is_go,
        # Three states, not two: see `preflight.GoNoGo.verdict`. `is_go` stays
        # what it was so nothing reading it changes meaning.
        "verdict": checklist.verdict,
        "all_from_the_book": checklist.all_from_the_book,
        "blockers": list(checklist.blockers),
        "unknowns": list(checklist.unknowns),
        "extrapolations": [
            {
                "where": entry.where,
                "what": entry.what,
                "detail": entry.detail,
                "conservative": entry.conservative,
                "summary": entry.summary,
            }
            for entry in checklist.extrapolations
        ],
        "airports": [
            {
                "airport": check.airport,
                "operation": check.operation,
                "elevation_ft": check.elevation_ft,
                "pressure_altitude_ft": check.pressure_altitude_ft,
                "density_altitude_ft": check.density_altitude_ft,
                "oat_c": check.oat_c,
                "weight_lb": check.weight_lb,
                "margin": check.margin,
                "passes": check.passes,
                # Magnetic, because that is the frame the runways are in and
                # the frame the numbers beside them were worked out in.
                "wind_from_deg": None if check.wind is None else check.wind.from_deg,
                "wind_speed_kt": None if check.wind is None else check.wind.speed_kt,
                "gust_kt": None if check.wind is None else check.wind.gust_kt,
                # Null where the field had no report at all, which is not the
                # same as a report the check could not judge -- that one comes
                # back with `passes: null` and says why.
                "weather": _weather_check_json(check.weather),
                "runways": [
                    {
                        "runway": runway.runway,
                        "surface": runway.surface,
                        "dry_grass_applied": runway.dry_grass_applied,
                        "end_used": runway.end_used,
                        "headwind_kt": runway.headwind_kt,
                        "crosswind_kt": runway.crosswind_kt,
                        "crosswind_from_right": runway.crosswind_from_right,
                        "crosswind_exceeds_demonstrated": (
                            runway.crosswind_exceeds_demonstrated
                        ),
                        "runway_available_ft": runway.runway_available_ft,
                        "ground_roll_ft": runway.ground_roll_ft,
                        "over_50ft_ft": runway.over_50ft_ft,
                        "required_ft": runway.required_ft,
                        "spare_ft": runway.spare_ft,
                        "passes": runway.passes,
                        "outside_envelope": runway.outside_envelope,
                        "extrapolated": runway.extrapolated,
                        "off_chart": [
                            {
                                "what": entry.what,
                                "detail": entry.detail,
                                "conservative": entry.conservative,
                            }
                            for entry in runway.off_chart
                        ],
                        "note": runway.note,
                    }
                    for runway in check.runways
                ],
            }
            for check in checklist.airports
        ],
        "fuel": {
            "fuel_on_board_gal": checklist.fuel.fuel_on_board_gal,
            "burn_gal": checklist.fuel.burn_gal,
            "landing_with_gal": checklist.fuel.landing_with_gal,
            "reserve_minutes": checklist.fuel.reserve_minutes,
            "reserve_required_gal": checklist.fuel.reserve_required_gal,
            "required_with_margin_gal": checklist.fuel.required_with_margin_gal,
            "margin": checklist.fuel.margin,
            "night": checklist.fuel.night,
            "spare_gal": checklist.fuel.spare_gal,
            "spare_minutes": checklist.fuel.spare_minutes,
            "passes": checklist.fuel.passes,
        },
    }


def _weather_check_json(check: pf.WeatherCheck | None) -> dict | None:
    if check is None:
        return None
    return {
        "visibility_sm": check.visibility_sm,
        "ceiling_ft_agl": check.ceiling_ft_agl,
        "ceiling_cover": check.ceiling_cover,
        "obscured": check.obscured,
        "sky_reported": check.sky_reported,
        "pattern_altitude_agl_ft": check.pattern_altitude_agl_ft,
        "required_ceiling_ft_agl": check.required_ceiling_ft_agl,
        "passes": check.passes,
        "reasons": list(check.reasons),
        "notes": list(check.notes),
        "summary": check.summary,
    }


# --- E6B -----------------------------------------------------------------
#
# The two standalone calculators in the right-hand bar. They take typed
# numbers rather than the route, so they are GETs with no request model, and
# like everything else here each is a direct translation of an engine call.


@app.get("/api/e6b/density-altitude")
def e6b_density_altitude(
    elevation_ft: float, oat_c: float, altimeter_inhg: float
) -> dict:
    """Pressure and density altitude for a field elevation, OAT and setting.

    The rules of thumb come back beside the real figures: a pilot checking
    this against the arithmetic in their head should see both.
    """
    try:
        pa = atm.pressure_altitude(elevation_ft, altimeter_inhg)
        da = atm.density_altitude(pa, oat_c)
        approx_pa = atm.pressure_altitude_approx(elevation_ft, altimeter_inhg)
        approx_da = atm.density_altitude_approx(pa, oat_c)
        isa_dev = atm.isa_deviation_c(pa, oat_c)
    except ValueError as exc:  # AboveModelCeiling and friends
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "pressure_altitude_ft": pa,
        "density_altitude_ft": da,
        "pressure_altitude_approx_ft": approx_pa,
        "density_altitude_approx_ft": approx_da,
        "isa_deviation_c": isa_dev,
    }


@app.get("/api/e6b/variation")
def e6b_variation(lat: float, lon: float, elevation_ft: float = 0.0) -> dict:
    """Magnetic variation at a point, positive east.

    Dated today, like the navlog's is: the field drifts about a tenth of a
    degree a year, so a fixed date would silently age.
    """
    if not -90.0 <= lat <= 90.0:
        return {"ok": False, "error": "Latitude must be between -90 and 90."}
    if not -180.0 <= lon <= 360.0:
        return {"ok": False, "error": "Longitude must be between -180 and 360."}
    decimal_year = mag.decimal_year_now()
    try:
        mag.check_validity(decimal_year)
        variation_deg = mag.variation(lat, lon, elevation_ft, decimal_year)
    except (OutsideModelValidity, AtGeographicPole) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "variation_deg": variation_deg,
        "decimal_year": decimal_year,
    }


@app.post("/api/e6b/weight-balance")
def e6b_weight_balance(request: WeightBalanceRequest) -> dict:
    """Gross weight and centre of gravity for a list of loading stations.

    A POST rather than a GET only because the loading is a list: like the
    other two calculators it takes typed numbers and knows nothing of the
    route.
    """
    try:
        loading = wb.compute(
            [
                wb.Station(
                    name=station.name,
                    weight_lb=station.weight_lb,
                    arm_in=station.arm_in,
                )
                for station in request.stations
            ]
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "gross_weight_lb": loading.gross_weight_lb,
        "total_moment_in_lb": loading.total_moment_in_lb,
        "cg_in": loading.cg_in,
        "moments_in_lb": [station.moment_in_lb for station in loading.stations],
    }


def _dataset_json(dataset: cur.Dataset, today: date) -> dict:
    return {
        "key": dataset.key,
        "label": dataset.label,
        "effective": dataset.effective.isoformat() if dataset.effective else None,
        "expires": dataset.expires.isoformat() if dataset.expires else None,
        "expired": dataset.expired(today),
        "days_remaining": dataset.days_remaining(today),
        "note": dataset.note,
    }


@app.get("/api/status")
def status() -> dict:
    today = datetime.now(tz=UTC).date()
    # Reported even when the database is missing: knowing the data is stale
    # is useful whether or not it loaded.
    data = [_dataset_json(d, today) for d in cur.datasets(today)]
    try:
        airport_count = apt.count()
        waypoint_count = apt.count_vfr_waypoints()
    except apt.AirportDatabaseMissing as exc:
        return {"ok": False, "error": str(exc), "data": data}
    return {
        "ok": True,
        "airports": airport_count,
        "vfr_waypoints": waypoint_count,
        "data": data,
    }


# --- raster charts -------------------------------------------------------

# One tile store per chart, kept for the life of the process: the decoded
# sectional is two hundred megabytes and takes a second to open, so it is
# opened once and then answers every tile from memory.
_tile_stores: dict[str, render.TileStore] = {}
_tile_stores_lock = threading.Lock()


def _tiles_url(chart: ch.Chart) -> str:
    """Where the map should ask for this chart's tiles.

    Pre-rendered, the whole pyramid is already under `data/`, which is
    mounted as static files -- so the tiles are served by Starlette with no
    Python in the path, and a deployed instance needs neither the GeoTIFFs
    nor the memory to decode them. With the charts themselves on disk the
    map goes through the render endpoint instead, which fills the same cache
    on demand and so is never short of a zoom level.
    """
    root = "/data/charts/tiles" if chart.prerendered else "/api/charts/tiles"
    return f"{root}/{chart.kind}/{chart.slug}/{{z}}/{{x}}/{{y}}.png"


def _chart_json(chart: ch.Chart, today: date) -> dict:
    dated = cur.Dataset(chart.key, chart.name, chart.effective, chart.expires, chart.note)
    return {
        "kind": chart.kind,
        "kind_label": chart.kind_label,
        "slug": chart.slug,
        "name": chart.name,
        "bounds": list(chart.bounds),
        "min_zoom": chart.min_zoom,
        "max_zoom": chart.max_zoom,
        "effective": chart.effective.isoformat() if chart.effective else None,
        "expires": chart.expires.isoformat() if chart.expires else None,
        "expired": dated.expired(today),
        "days_remaining": dated.days_remaining(today),
        "note": chart.note,
        "prerendered": chart.prerendered,
        "tiles": _tiles_url(chart),
    }


@app.get("/api/charts")
def list_charts() -> dict:
    """Every chart this instance can draw, with its footprint and its dates.

    From the GeoTIFFs under data/charts when they are there, and from the
    tile manifest when they are not -- which is the deployed case.
    """
    today = datetime.now(tz=UTC).date()
    return {"ok": True, "charts": [_chart_json(chart, today) for chart in ch.available()]}


def _store_for(kind: str, slug: str) -> render.TileStore:
    key = f"{kind}/{slug}"
    with _tile_stores_lock:
        store = _tile_stores.get(key)
        if store is None:
            chart = ch.find(kind, slug)
            if chart is None:
                raise HTTPException(status_code=404, detail=f"no chart {key}")
            try:
                store = _tile_stores[key] = chart.store()
            except ch.NoChartFile as exc:
                # Pre-rendered tiles are served from /data; nothing to render.
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        return store


@app.get("/api/charts/tiles/{kind}/{slug}/{z}/{x}/{y}.png")
def chart_tile(kind: str, slug: str, z: int, x: int, y: int) -> Response:
    """One Web Mercator tile of a chart, rendered on first request and kept."""
    if not 0 <= z <= 20 or not 0 <= x < 2**z or not 0 <= y < 2**z:
        raise HTTPException(status_code=404, detail="no such tile")
    data = _store_for(kind, slug).get(z, x, y)
    # Tiles change only when the chart file does, which is once per edition.
    return Response(
        data, media_type=render.PNG_MEDIA_TYPE, headers={"Cache-Control": "public, max-age=86400"}
    )


# --- static files --------------------------------------------------------


@app.middleware("http")
async def no_cache_the_ui(request, call_next):
    """Never let the browser cache the UI source while developing.

    `index.html` and `app.js` have to move together: the table's headers live
    in one and its cells in the other. Cache one and not the other and every
    column shifts under its heading, which looks like a rendering bug and is
    really a stale file. The basemap and airport data are left cacheable --
    they are large and change only when the data pipeline is re-run.
    """
    response = await call_next(request)
    if request.url.path in ("/", "/index.html") or request.url.path.endswith(
        (".js", ".css")
    ):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    elif request.url.path.startswith("/data/charts/tiles/"):
        # The opposite case. A chart tile changes only when the edition does,
        # once every 56 days, and a map pan asks for dozens of them. Without
        # this each one is a conditional request to the server on every pan.
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response


# The basemap is served from data/ so that the same files are baked into the
# PWA bundle later without being duplicated here.
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


app.mount("/", StaticFiles(directory=UI_DIR), name="ui")
