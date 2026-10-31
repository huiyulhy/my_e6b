"""Fetching the flights that flew a city pair, and their tracks, from OpenSky.

The network half of `engine/tracks.py`, on the same split as
`server/wx_surface.py` and `server/notams.py`: this reaches the outside
world and knows nothing about what a track means.

**Credentials are optional.** OpenSky serves these endpoints anonymously,
on 400 credits a day per endpoint family, bucketed by caller IP. An account
raises that to 4,000 and is worth having, but nothing here depends on it.
With one, create an API client on the account page at opensky-network.org
and set:

    export OPENSKY_CLIENT_ID=...
    export OPENSKY_CLIENT_SECRET=...

(OAuth2 client credentials; passwords stopped working in March 2026.) The
pair is exchanged for a bearer token, which lasts about thirty minutes and
is kept in memory until shortly before it expires. With nothing set, the
requests go out unauthenticated and the result says so, so the UI can name
the smaller allowance when the credits run out.

**Three calls per route, in two stages.** The flights-by-departure endpoint
returns everything that left a field in a window of at most two days, so ten
days is five windows -- fetched concurrently, as the weather fetcher does.
Those records say where each flight ended, which is how the pair is found;
then one tracks call per chosen flight, newest first, stopping once `limit`
of them have passed the Class A test. The search is capped at `MAX_TRACK_FETCHES`
because every track is a request against a daily credit allowance that is
small on the free tier, and a pair with no VFR traffic should not spend the
day's credits proving it.

**Yesterday is the newest day there is.** OpenSky fills its flights tables by
a nightly batch, so a flight that landed this afternoon is not in them until
tomorrow. The window therefore ends at the start of today (UTC) and runs the
requested number of whole days back from there: asking about today would
spend credits on a window that is empty by construction.

**Cached on disk** under `.cache/opensky/`, like the other fetchers. A window
that is wholly in the past is final once the batch has run, so it is kept
for a day; a track, once seen, never changes, and is kept for a week.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from engine import tracks as tr

__all__ = [
    "CLIENT_ID_ENV",
    "CLIENT_SECRET_ENV",
    "DEFAULT_DAYS",
    "DEFAULT_LIMIT",
    "MAX_DAYS",
    "MAX_LIMIT",
    "MAX_TRACK_FETCHES",
    "FlightsUnavailable",
    "RouteFlights",
    "credentials_configured",
    "fetch_route_flights",
]

logger = logging.getLogger(__name__)

OPENSKY_API = "https://opensky-network.org/api"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
)
CLIENT_ID_ENV = "OPENSKY_CLIENT_ID"
CLIENT_SECRET_ENV = "OPENSKY_CLIENT_SECRET"

USER_AGENT = "my_e6b VFR planner (+https://github.com/huiyulhy/my_e6b)"
TIMEOUT_S = 30.0
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "opensky"

# Seconds. A finished day's flights are final once the nightly batch has run;
# a track, once recorded, never changes.
TTL_FLIGHTS = 24 * 3600
TTL_TRACKS = 7 * 24 * 3600

# How many days back to look, and how far the UI may push it. OpenSky keeps
# tracks for thirty days; the hard cap sits under that so a requested track
# is never one the service has already discarded.
DEFAULT_DAYS = 10
MAX_DAYS = 28
# The longest window the flights-by-airport endpoints accept.
WINDOW = timedelta(days=2)

# How many flights to draw, and how many tracks may be asked for to find
# them. Each track is one request against the day's credits, and a pair
# whose traffic is all airline would otherwise spend them all failing the
# Class A test one flight at a time.
DEFAULT_LIMIT = 10
MAX_LIMIT = 20
MAX_TRACK_FETCHES = 30

# Renew the token this long before OpenSky says it expires, so a request
# never goes out on a token with seconds left.
_TOKEN_SLACK_S = 60.0


class FlightsUnavailable(RuntimeError):
    """OpenSky could not be reached, or is not configured."""

    def __init__(self, message: str, *, needs_credentials: bool = False) -> None:
        super().__init__(message)
        self.needs_credentials = needs_credentials


@dataclass(frozen=True)
class RouteFlights:
    """What one search found, and what it had to look at to find it."""

    departure: str
    arrival: str
    begin: datetime
    end: datetime
    tracks: tuple[tr.FlightTrack, ...]
    # Every flight between the pair in the window, before any filtering.
    matched: int
    # Those with a tail-number callsign, newest first -- the candidates.
    candidates: int
    # How many tracks were asked for, and how many of those were refused for
    # reaching Class A or for being too thin to draw.
    tracks_fetched: int
    rejected_class_a: int
    rejected_thin: int
    # Candidates not looked at because the fetch cap was reached first.
    unexamined: int
    # Windows that could not be fetched. A partial answer is labelled, never
    # passed off as the whole ten days.
    failed_windows: tuple[str, ...] = field(default_factory=tuple)
    # Whether the requests went out without an API client, on the smaller
    # anonymous allowance.
    anonymous: bool = False

    @property
    def complete(self) -> bool:
        return not self.failed_windows and self.unexamined == 0


# --- credentials and the token ----------------------------------------------


def _credentials() -> tuple[str, str]:
    return (
        os.environ.get(CLIENT_ID_ENV, "").strip(),
        os.environ.get(CLIENT_SECRET_ENV, "").strip(),
    )


def credentials_configured() -> bool:
    """Whether OpenSky can be called at all, without calling it."""
    client_id, secret = _credentials()
    return bool(client_id and secret)


_token_lock = threading.Lock()
_token: tuple[str, float] | None = None  # (bearer, expires_at epoch seconds)


def _token_error(exc: urllib.error.HTTPError) -> str:
    """The OAuth2 error from a token rejection, as `: reason`, or empty.

    Read before anything else touches the response, because an HTTPError's
    body can only be read once and is gone if it is not taken here.
    """
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 -- diagnostics must never mask the real error
        return ""
    if not body:
        return ""
    try:
        answer = json.loads(body)
    except ValueError:
        return f": {body[:200]}"
    code = answer.get("error")
    description = answer.get("error_description")
    detail = " - ".join(part for part in (code, description) if part)
    return f": {detail}" if detail else ""


def _bearer(*, refresh: bool = False) -> str:
    """The current access token, fetched or renewed as needed."""
    global _token
    with _token_lock:
        if not refresh and _token and time.time() < _token[1] - _TOKEN_SLACK_S:
            return _token[0]
        client_id, secret = _credentials()
        if not (client_id and secret):
            raise FlightsUnavailable(
                f"Flown routes need an OpenSky API client: set {CLIENT_ID_ENV} and "
                f"{CLIENT_SECRET_ENV} before starting the server",
                needs_credentials=True,
            )
        body = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": secret,
            }
        ).encode("ascii")
        request = urllib.request.Request(
            TOKEN_URL,
            data=body,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                answer = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The token service says *why* in the body, as OAuth2 requires:
            # `invalid_client` is a wrong id or secret, `unauthorized_client`
            # a client that exists but may not use this grant. Without it
            # every rejection reads the same and there is nothing to act on.
            reason = _token_error(exc)
            if exc.code in (400, 401, 403):
                raise FlightsUnavailable(
                    f"OpenSky rejected the API client ({exc.code}{reason}); check "
                    f"{CLIENT_ID_ENV} and {CLIENT_SECRET_ENV} against the client "
                    f"created on your OpenSky account page",
                    needs_credentials=True,
                ) from exc
            raise FlightsUnavailable(
                f"OpenSky token service answered {exc.code}{reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise FlightsUnavailable(f"could not reach OpenSky's token service: {exc}") from exc
        token = answer.get("access_token") if isinstance(answer, dict) else None
        if not token:
            raise FlightsUnavailable("OpenSky's token service answered without a token")
        expires_in = float(answer.get("expires_in", 1800) or 1800)
        _token = (str(token), time.time() + expires_in)
        return _token[0]


# --- the GET, with its cache --------------------------------------------------


def _cache_path(url: str) -> Path:
    from hashlib import sha256

    digest = sha256(url.encode("utf-8")).hexdigest()[:16]
    return CACHE / f"{digest}.json"


def _get_json(url: str, ttl_s: float, *, refresh: bool = False) -> Any:
    """GET as the configured client, serving from disk while fresh.

    The same shape as the other fetchers' `_get_json`, separate because the
    authentication is a bearer token that can lapse mid-session: a 401 is
    answered once by renewing the token and asking again, which the key-based
    fetchers never need.
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

    body = _get_authenticated(url)
    try:
        payload = json.loads(body) if body.strip() else None
    except ValueError as exc:
        raise FlightsUnavailable("OpenSky returned malformed JSON") from exc

    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"fetched_at": time.time(), "payload": payload}, handle)
        tmp.replace(path)
    except OSError:
        pass  # A cache we cannot write is a slower program, not a broken one.
    return payload


def _get_authenticated(url: str) -> str:
    """GET with a bearer token when there is a client, anonymously otherwise."""
    signed = credentials_configured()
    for attempt in (1, 2):
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if signed:
            headers["Authorization"] = f"Bearer {_bearer(refresh=attempt == 2)}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # A token that lapsed between the check and the request: renew
            # once and retry. A second 401 is the client itself.
            if exc.code == 401 and attempt == 1 and signed:
                continue
            # OpenSky answers 404 for "no flights in this window", which is
            # an empty list and not a failure.
            if exc.code == 404:
                return ""
            if exc.code in (401, 403):
                if not signed:
                    raise FlightsUnavailable(
                        f"OpenSky refused the anonymous request ({exc.code}); set "
                        f"{CLIENT_ID_ENV} and {CLIENT_SECRET_ENV} from an API client "
                        f"on your OpenSky account",
                        needs_credentials=True,
                    ) from exc
                raise FlightsUnavailable(
                    f"OpenSky refused the request ({exc.code}); the API client may "
                    f"lack access to this endpoint"
                ) from exc
            if exc.code == 429:
                allowance = (
                    "the anonymous allowance of 400 credits a day is used up; an "
                    f"OpenSky API client ({CLIENT_ID_ENV}, {CLIENT_SECRET_ENV}) has ten times that"
                    if not signed
                    else "the day's credits are used up"
                )
                raise FlightsUnavailable(f"OpenSky rate limit reached (429): {allowance}") from exc
            raise FlightsUnavailable(f"OpenSky answered {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FlightsUnavailable(f"could not reach OpenSky: {exc}") from exc
    raise FlightsUnavailable("OpenSky kept refusing the token")  # unreachable


# --- the search ------------------------------------------------------------------


def _windows(begin: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Two-day windows covering [begin, end], the last one reaching `end`."""
    windows = []
    start = begin
    while start < end:
        stop = min(start + WINDOW, end)
        windows.append((start, stop))
        start = stop
    return windows


def _fetch_departures(
    airport: str, window: tuple[datetime, datetime], *, refresh: bool
) -> list[tr.Flight]:
    begin, end = window
    query = urllib.parse.urlencode(
        {
            "airport": airport,
            "begin": int(begin.timestamp()),
            "end": int(end.timestamp()),
        }
    )
    payload = _get_json(f"{OPENSKY_API}/flights/departure?{query}", TTL_FLIGHTS, refresh=refresh)
    try:
        return tr.parse_flights(payload)
    except tr.UnrecognisedPayload as exc:
        raise FlightsUnavailable(f"OpenSky's flight list was unreadable: {exc}") from exc


def _fetch_track(flight: tr.Flight, *, refresh: bool) -> tr.FlightTrack:
    query = urllib.parse.urlencode(
        {
            "icao24": flight.icao24,
            "time": int(flight.midpoint.timestamp()),
        }
    )
    payload = _get_json(f"{OPENSKY_API}/tracks/all?{query}", TTL_TRACKS, refresh=refresh)
    try:
        return tr.parse_track(payload, flight)
    except tr.UnrecognisedPayload as exc:
        raise FlightsUnavailable(f"OpenSky's track was unreadable: {exc}") from exc


def fetch_route_flights(
    departure: str,
    arrival: str,
    *,
    days: int = DEFAULT_DAYS,
    limit: int = DEFAULT_LIMIT,
    now: datetime | None = None,
    refresh: bool = False,
) -> RouteFlights:
    """The most recent flights from `departure` to `arrival` that look VFR.

    Both identifiers are ICAO codes, as OpenSky files them. Raises
    `FlightsUnavailable` when nothing at all could be fetched; a window that
    failed among others that succeeded is recorded on the result instead.
    """
    departure = departure.strip().upper()
    arrival = arrival.strip().upper()
    if not departure or not arrival or departure == arrival:
        raise FlightsUnavailable("a route needs two different airports")
    days = max(1, min(int(days), MAX_DAYS))
    limit = max(1, min(int(limit), MAX_LIMIT))

    # Whole UTC days, ending where today begins: OpenSky has nothing for
    # today until tonight's batch, so a window into it is spent for nothing.
    now = now or datetime.now(tz=UTC)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    begin = end - timedelta(days=days)
    windows = _windows(begin, end)

    flights: list[tr.Flight] = []
    failed: list[str] = []
    last_error: FlightsUnavailable | None = None
    with ThreadPoolExecutor(max_workers=len(windows)) as pool:
        futures = [
            pool.submit(_fetch_departures, departure, window, refresh=refresh) for window in windows
        ]
        for window, future in zip(windows, futures, strict=True):
            try:
                flights.extend(future.result())
            except FlightsUnavailable as exc:
                logger.warning("OpenSky departures %s %s: %s", departure, window[0].date(), exc)
                failed.append(f"{window[0].date().isoformat()} to {window[1].date().isoformat()}")
                last_error = exc
    if last_error is not None and len(failed) == len(windows):
        # Nothing at all came back. The reason is the same for every window
        # -- credentials, credits, network -- so the last one speaks for all.
        raise last_error

    matched = [f for f in flights if (f.arrival or "").upper() == arrival]
    # Every candidate, so the ones past the fetch cap can be counted; only the
    # first MAX_TRACK_FETCHES are ever asked about.
    all_candidates = tr.most_recent(flights, arrival=arrival, limit=len(flights) or 1)
    candidates = all_candidates[:MAX_TRACK_FETCHES]

    kept: list[tr.FlightTrack] = []
    fetched = rejected_class_a = rejected_thin = 0
    examined = 0
    for flight in candidates:
        if len(kept) >= limit:
            break
        examined += 1
        fetched += 1
        try:
            track = _fetch_track(flight, refresh=refresh)
        except FlightsUnavailable as exc:
            logger.warning("OpenSky track %s: %s", flight.icao24, exc)
            # A missing track is OpenSky's gap, not a verdict on the flight;
            # counted as thin so the summary still adds up.
            rejected_thin += 1
            continue
        if len(track.points) < tr.MIN_TRACK_POINTS:
            rejected_thin += 1
            continue
        if not tr.flew_below_class_a(track):
            rejected_class_a += 1
            continue
        kept.append(track)

    return RouteFlights(
        departure=departure,
        arrival=arrival,
        begin=begin,
        end=end,
        tracks=tuple(kept),
        matched=len({(f.icao24, f.first_seen) for f in matched}),
        candidates=len(all_candidates),
        tracks_fetched=fetched,
        rejected_class_a=rejected_class_a,
        rejected_thin=rejected_thin,
        # Candidates the cap kept from being looked at. When `limit` tracks
        # were found the rest were not needed, which is not the same thing.
        unexamined=0 if len(kept) >= limit else len(all_candidates) - examined,
        failed_windows=tuple(failed),
        anonymous=not credentials_configured(),
    )
