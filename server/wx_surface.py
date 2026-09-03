"""Fetch surface weather for one field: AWC METAR and TAF, plus the model hour.

This is the network half of the weather tier. It lives in `server/` and not in
`engine/` because `engine/` touches no network -- the invariant that lets the
same engine run under Pyodide in airplane mode. Everything here is I/O,
caching and error handling; every decision about *which* number to believe is
in `engine/weather.py`, which is pure and tested offline.

**Three sources, fetched concurrently.** A surface request for a future time
needs a METAR, a TAF and an hour of model output, and they are independent, so
they go out together rather than one after another. `urllib` is synchronous, so
the concurrency is a small thread pool; FastAPI runs the calling endpoint in a
threadpool of its own because it is declared `def` rather than `async def`.

**Stdlib only.** No `httpx`, no `requests`. `server/` is the dev shim that
never ships, and it would be the only shipped-tier dependency that is never
actually shipped. Two GETs and a JSON decode do not need a library.

**Responses are cached on disk** under `.cache/wx/`, the same convention
`tools/build_*.py` uses for FAA downloads. Planning is an edit-and-look loop --
drag a waypoint, read the numbers again -- and without a cache every keystroke
would be three HTTP requests to a public service that asks callers not to
hammer it. The TTLs match how often the products actually change: a METAR is
issued hourly, a TAF every six hours, model output every hour.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from engine import airports as apt
from engine import aloft as al
from engine import weather as wx
from engine.geo import LatLon
from engine.weather import SurfaceWeather, WeatherUnavailable

__all__ = ["WeatherUnavailable", "fetch_aloft", "fetch_surface"]

AWC = "https://aviationweather.gov/api/data"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# A courteous identifier. Public services are entitled to know who is calling,
# and an unidentified client is the first thing a rate limiter drops.
USER_AGENT = "my_e6b VFR planner (+https://github.com/huiyulhy/my_e6b)"

TIMEOUT_S = 15.0
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "wx"

# Seconds. A METAR is hourly with specials between; a TAF is issued four times
# a day and amended; model output updates hourly at best.
TTL_METAR = 5 * 60
TTL_TAF = 30 * 60
TTL_MODEL = 30 * 60

# How far to look for a neighbouring field's TAF, and how big a field has to be
# to be likely to publish one. Only around 600 US airports issue a TAF, and
# they are the towered ones with long runways.
NEAREST_TAF_RADIUS_NM = 75.0
NEAREST_TAF_MIN_RUNWAY_FT = 5000.0
NEAREST_TAF_CANDIDATES = 6


@dataclass(frozen=True)
class _Cached:
    payload: Any
    fetched_at: float


def _cache_path(url: str) -> Path:
    """A readable filename per URL, so the cache can be inspected by eye."""
    from hashlib import sha256

    digest = sha256(url.encode("utf-8")).hexdigest()[:16]
    return CACHE / f"{digest}.json"


def _get_json(url: str, ttl_s: float, *, refresh: bool = False) -> Any:
    """GET and decode, serving from the disk cache while it is fresh.

    Raises `WeatherUnavailable` on any transport or decode failure. It never
    returns a partial or stale-but-unlabelled result: a caller that cannot
    reach a source needs to know that, because the alternative is a runway
    length computed from yesterday's air.
    """
    path = _cache_path(url)
    if not refresh and path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                cached = json.load(handle)
            if time.time() - float(cached["fetched_at"]) < ttl_s:
                return cached["payload"]
        except (OSError, ValueError, KeyError):
            pass  # A corrupt cache entry is simply a miss.

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WeatherUnavailable(f"could not reach {_host(url)}: {exc}") from exc

    # AWC answers a station that publishes no TAF with an empty body rather
    # than an empty list. That is "there is no TAF here", which is a normal
    # answer for most of the fields this program plans to -- not a transport
    # failure, and it must not be reported as one.
    if not body.strip():
        payload: Any = []
    else:
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise WeatherUnavailable(f"{_host(url)} returned malformed JSON") from exc

    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"fetched_at": time.time(), "payload": payload}, handle)
        tmp.replace(path)  # atomic, so a crash cannot leave a half-written entry
    except OSError:
        pass  # A cache we cannot write is a slower program, not a broken one.

    return payload


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc or url


# --- the three sources ----------------------------------------------------


def _metar_url(ids: str, hours: int | None = None) -> str:
    query = {"ids": ids, "format": "json"}
    if hours is not None:
        query["hours"] = str(hours)
    return f"{AWC}/metar?{urllib.parse.urlencode(query)}"


def _taf_url(ids: str) -> str:
    return f"{AWC}/taf?{urllib.parse.urlencode({'ids': ids, 'format': 'json'})}"


def _model_url(position: LatLon, target: datetime) -> str:
    """One day either side of the target, hourly, in aviation units.

    `pressure_msl` rather than `surface_pressure`: an altimeter setting is a
    sea-level-reduced pressure, and station pressure would double-count the
    field elevation downstream. `wind_speed_unit=kn` so nothing here has to
    remember to convert.
    """
    day = target.astimezone(UTC).date()
    query = {
        "latitude": f"{position.lat:.4f}",
        "longitude": f"{position.lon:.4f}",
        "hourly": (
            "temperature_2m,pressure_msl,wind_speed_10m,"
            "wind_direction_10m,wind_gusts_10m"
        ),
        "wind_speed_unit": "kn",
        "temperature_unit": "celsius",
        "timezone": "GMT",
        "start_date": day.isoformat(),
        "end_date": (day + timedelta(days=1)).isoformat(),
        "models": "gfs_seamless",
    }
    return f"{OPEN_METEO}?{urllib.parse.urlencode(query)}"


def _aloft_url(position: LatLon, target: datetime, levels: tuple[int, ...]) -> str:
    """The same forecast as `_model_url`, asked on pressure levels.

    A separate request rather than more `hourly=` names on the surface one:
    the two have different lifetimes in the cache and different failure
    consequences. A field with no wind aloft still has a usable altimeter
    setting, and losing the go/no-go because the cruise wind was unavailable
    would be the wrong trade.
    """
    day = target.astimezone(UTC).date()
    query = {
        "latitude": f"{position.lat:.4f}",
        "longitude": f"{position.lon:.4f}",
        "hourly": ",".join(al.hourly_fields(levels)),
        "wind_speed_unit": "kn",
        "temperature_unit": "celsius",
        "timezone": "GMT",
        "start_date": day.isoformat(),
        "end_date": (day + timedelta(days=1)).isoformat(),
        "models": "gfs_seamless",
    }
    return f"{OPEN_METEO}?{urllib.parse.urlencode(query)}"


def fetch_aloft(
    position: LatLon,
    target: datetime | None = None,
    *,
    ceiling_ft: float = 14000.0,
    refresh: bool = False,
) -> al.AloftForecast:
    """Winds and temperatures aloft over one point, for now or for a time.

    Keyed by position, not by identifier: the wind at cruise belongs to the
    piece of sky the leg crosses, and asking about the departure airport would
    reproduce the very thing the FD product gets wrong.

    Raises `WeatherUnavailable` when the model cannot be reached, or when it
    answers with nothing usable for this hour -- an empty forecast would read
    as "calm and standard", which is a claim about the day rather than the
    absence of one.
    """
    target = datetime.now(tz=UTC) if target is None else _as_utc(target)
    levels = al.levels_up_to(ceiling_ft)
    payload = _get_json(_aloft_url(position, target, levels), TTL_MODEL, refresh=refresh)
    forecast = al.parse_aloft(payload, target, levels=levels)
    if forecast is None:
        raise WeatherUnavailable(
            f"the model returned no hour near {target:%Y-%m-%d %H:%MZ} "
            f"for {position.lat:.3f}, {position.lon:.3f}"
        )
    return forecast


def _nearest_taf_candidates(position: LatLon) -> list[apt.Airport]:
    """Sizeable fields near a station that publishes no TAF of its own.

    Reuses `airports.near`, which already does the bounding-box-then-geodesic
    narrowing, so this is a database read rather than a second network call.
    """
    try:
        nearby = apt.near(
            position,
            NEAREST_TAF_RADIUS_NM,
            limit=40,
            min_runway_ft=NEAREST_TAF_MIN_RUNWAY_FT,
        )
    except apt.AirportDatabaseMissing:
        return []
    return [airport for airport in nearby if airport.icao][:NEAREST_TAF_CANDIDATES]


# --- the public call ------------------------------------------------------


def fetch_surface(
    ident: str, target: datetime | None = None, *, refresh: bool = False
) -> SurfaceWeather:
    """Surface weather at one field, for now or for a target time.

    `target` is read as UTC when naive. A target in the past is answered from
    the current METAR rather than refused, since a plan written for this
    morning is still a plan; a target beyond the model window comes back with
    the fields nobody could supply left as `None`.

    Raises `AirportUnknown` when the identifier is not in the airport
    database, and `WeatherUnavailable` when no source could be reached at all.
    A source that is merely *empty* -- a field with no TAF, a station not
    reporting -- is not an error; it is handled by falling back and noted in
    the result.
    """
    ident = ident.strip().upper()
    airport = apt.find(ident)
    if airport is None:
        raise AirportUnknown(f"{ident} is not in the airport database")

    now = datetime.now(tz=UTC)
    target = now if target is None else _as_utc(target)
    station = (airport.icao or airport.ident).upper()
    ahead = (target - now).total_seconds() / 60.0
    wants_forecast = ahead > 60.0

    # The model is fetched even for a request about right now. Most fields a
    # light aircraft uses are part-time: KSQL stops reporting overnight, and
    # its "latest" METAR can be three hours old at 2am. Without a model to
    # fall back to, a stale observation is correctly rejected and the pilot
    # gets nothing at all. One extra concurrent request buys an answer at
    # every field at every hour.
    jobs: dict[str, tuple] = {
        "metar": (_metar_url(station), TTL_METAR),
        "model": (_model_url(airport.position, target), TTL_MODEL),
    }
    if wants_forecast:
        jobs["taf"] = (_taf_url(station), TTL_TAF)

    payloads, failures = _fetch_all(jobs, refresh=refresh)
    if len(failures) == len(jobs):
        # Everything failed -- almost certainly no network at all.
        raise WeatherUnavailable("; ".join(sorted(set(failures.values()))))

    metar = wx.parse_metar(payloads.get("metar"))
    taf = wx.parse_taf(payloads.get("taf"), target) if wants_forecast else None
    model = wx.parse_model_surface(payloads.get("model"), target, station=station)

    taf_station: str | None = station if taf is not None else None
    taf_distance_nm: float | None = None

    # The station has no TAF of its own, or none covering the target. Borrow
    # the nearest one that does, and say so in the result.
    if wants_forecast and taf is None:
        borrowed = _borrow_taf(airport.position, target, refresh=refresh)
        if borrowed is not None:
            taf, taf_station, taf_distance_nm = borrowed

    resolved = wx.resolve_surface(
        station=station,
        target=target,
        now=now,
        metar=metar,
        taf=taf,
        model=model,
        taf_station=taf_station,
        taf_distance_nm=taf_distance_nm,
    )

    if failures:
        note = "; ".join(f"{name} unavailable ({why})" for name, why in sorted(failures.items()))
        resolved = _with_note(resolved, note)
    return resolved


def _fetch_all(jobs: dict[str, tuple], *, refresh: bool) -> tuple[dict, dict]:
    """Run every job concurrently; return what succeeded and why the rest did not.

    One source being down must not take the others with it. A missing TAF
    still leaves a usable METAR, and saying so beats failing the whole request.
    """
    payloads: dict[str, Any] = {}
    failures: dict[str, str] = {}
    if not jobs:
        return payloads, failures

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            name: pool.submit(_get_json, url, ttl, refresh=refresh)
            for name, (url, ttl) in jobs.items()
        }
        for name, future in futures.items():
            try:
                payloads[name] = future.result()
            except WeatherUnavailable as exc:
                failures[name] = str(exc)
    return payloads, failures


def _borrow_taf(
    position: LatLon, target: datetime, *, refresh: bool
) -> tuple[SurfaceWeather, str, float] | None:
    """The nearest neighbouring field whose TAF covers `target`.

    Candidates are requested in a single call -- the AWC endpoint takes a
    comma-separated list -- and the nearest one that actually answers wins.
    """
    from engine.geo import distance_nm

    candidates = _nearest_taf_candidates(position)
    if not candidates:
        return None

    ids = ",".join(airport.icao for airport in candidates)
    try:
        payload = _get_json(_taf_url(ids), TTL_TAF, refresh=refresh)
    except WeatherUnavailable:
        return None
    if not isinstance(payload, list):
        return None

    by_station = {
        str(entry.get("icaoId") or "").upper(): entry
        for entry in payload
        if isinstance(entry, dict)
    }
    for airport in candidates:  # already nearest-first
        entry = by_station.get(airport.icao.upper())
        if entry is None:
            continue
        forecast = wx.parse_taf([entry], target)
        if forecast is not None:
            return forecast, airport.icao.upper(), distance_nm(position, airport.position)
    return None


def _with_note(report: SurfaceWeather, note: str) -> SurfaceWeather:
    from dataclasses import replace

    return replace(report, notes=tuple(dict.fromkeys((*report.notes, note))))


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


class AirportUnknown(LookupError):
    """The identifier is not in the airport database."""
