"""Fetching NOTAMs for a route from SkyLink, through RapidAPI.

The network half of `engine/notam.py`, on the same split as
`server/wx_surface.py`: this reaches the outside world and knows nothing about
whether a NOTAM matters.

**Credentials.** One RapidAPI key with a SkyLink subscription, read from the
environment rather than committed:

    export RAPIDAPI_KEY=...

With none set, `fetch_route` raises `NotamsUnavailable` saying so. An empty
NOTAM list is the most dangerous thing this module could return, since it
reads as "nothing to report", so a missing key is never allowed to look like
one.

**Asking by identifier.** SkyLink takes an ICAO code, not a position, so the
route is turned into the identifiers along it first -- and both kinds are
needed. A closed runway is filed against the aerodrome, but a TFR, an MOA or
an airspace closure is filed against the ARTCC whose airspace it sits in.

**Finding the identifiers.** The aerodromes come from the local airport
database, searched in a chain of circles laid along the route. Circles of
radius R every R nautical miles cover everything within R x sqrt(3)/2 of the
track -- the thin spot is halfway between two centres -- so the search radius
is set from the corridor width rather than equal to it. Searching 20 nm
circles every 20 nm would leave scalloped gaps, and an aerodrome 19 nm off
track halfway between two samples would never be asked about.

**The budget.** The free tier is 1,000 requests a month and each identifier
is one request, so a route is capped at `MAX_DESIGNATORS`. The route's own
fields are asked about first and the centres next, so the cap can only ever
drop outlying aerodromes -- never the destination, and never the centre where
the TFRs are. Anything cut is reported as an incomplete search.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine import notam as nt
from engine.geo import LatLon, inverse

__all__ = [
    "MAX_DESIGNATORS",
    "NotamsUnavailable",
    "RouteNotams",
    "credentials_configured",
    "fetch_route",
]

# SkyLink, through RapidAPI. One key, sent as a header; the host header is
# how RapidAPI routes the request to the right product.
SKYLINK_HOST = "skylink-api.p.rapidapi.com"
SKYLINK_API = f"https://{SKYLINK_HOST}/v3/notams"
RAPIDAPI_KEY_ENV = "RAPIDAPI_KEY"

USER_AGENT = "my_e6b VFR planner (+https://github.com/huiyulhy/my_e6b)"
TIMEOUT_S = 20.0
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "notams"

# Seconds. NOTAMs are issued continuously but a briefing five minutes old is
# still the briefing; shorter than this and replanning a route would spend the
# monthly allowance asking the same airports the same question.
TTL_S = 10 * 60

# How many identifiers one route may ask SkyLink about. Each is one request
# against a 1,000-a-month free tier, so this is the budget line: at the cap a
# month is 25 briefings. Beyond it the search is cut short and says so, rather
# than burning a month's allowance on one long cross-country.
MAX_DESIGNATORS = 40

# The most sample points one route may be broken into. A 1,000 nm route at the
# spacing below is well inside this; the cap is a backstop against a route
# with a mistyped waypoint on the other side of the world.
MAX_QUERY_POINTS = 40

# Circles of radius R spaced R apart cover everything within R*sqrt(3)/2 of
# the track: the thinnest point of the chain is halfway between two centres,
# where the two circles cross. Inverting that gives the radius a corridor
# needs.
_CHAIN_COVERAGE = math.sqrt(3.0) / 2.0


class NotamsUnavailable(RuntimeError):
    """SkyLink could not be reached, or is not configured.

    Raised rather than returning an empty list, and the distinction matters
    more here than anywhere else in this program: "no NOTAMs" and "no NOTAM
    service" look identical on a briefing page and mean opposite things.
    """


@dataclass(frozen=True)
class RouteNotams:
    """Everything one route's queries returned, and what they asked about.

    The identifiers are carried so the briefing can say what was actually
    searched. A pilot told "3 NOTAMs" is owed the difference between three
    found across every field on the route and three found before the rest of
    the queries failed.
    """

    notams: tuple[nt.Notam, ...]
    designators: tuple[str, ...]  # the aerodromes and centres asked about
    failed: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.failed


def credentials_configured() -> bool:
    """Whether SkyLink can be called at all, without calling it."""
    return bool(os.environ.get(RAPIDAPI_KEY_ENV))


def fetch_route(
    positions: list[LatLon] | tuple[LatLon, ...],
    *,
    corridor_nm: float = nt.DEFAULT_CORRIDOR_NM,
    priority: tuple[str, ...] = (),
    refresh: bool = False,
) -> RouteNotams:
    """Every NOTAM filed against an identifier along a route, de-duplicated.

    `positions` is the route, in order. `priority` is the route's own
    aerodromes -- departure, stops, destination -- which are asked about
    before anything else so that the budget cap can never cut them.

    An identifier that fails is recorded rather than raised on: one
    unreachable query out of seventeen is a partial briefing, which is worth
    showing as long as it is labelled. Every query failing raises.
    """
    if len(positions) < 2:
        raise NotamsUnavailable("a route needs at least two points")
    if not credentials_configured():
        raise NotamsUnavailable(
            f"NOTAMs need a SkyLink subscription on RapidAPI: set "
            f"{RAPIDAPI_KEY_ENV} before starting the server"
        )

    wanted = _designators(tuple(positions), corridor_nm, priority)
    asked, skipped = wanted[:MAX_DESIGNATORS], wanted[MAX_DESIGNATORS:]

    merged: dict[str, nt.Notam] = {}
    failed: list[str] = []
    answered = 0
    for designator in asked:
        try:
            payload = _get_json(
                f"{SKYLINK_API}/{urllib.parse.quote(designator)}", refresh=refresh
            )
            notams = nt.parse_skylink(payload)
        except NotamsUnavailable as exc:
            failed.append(f"{designator}: {exc}")
            continue
        except nt.UnrecognisedPayload as exc:
            # A reply that could not be read is an outage for that
            # identifier, never "no NOTAMs there".
            failed.append(f"{designator}: SkyLink sent an unreadable reply ({exc})")
            continue
        answered += 1
        for notam in notams:
            # First wins: the same NOTAM filed against a field and returned
            # again under its centre is one NOTAM.
            merged.setdefault(notam.key, notam)

    # Every identifier that was asked about failed. Checked on its own count,
    # not by scanning `failed`: the budget notice below lands in that list
    # too, and must not be able to turn a total outage into an empty list.
    if asked and answered == 0:
        raise NotamsUnavailable(f"no SkyLink query succeeded: {failed[0]}")

    if skipped:
        failed.append(
            f"{len(skipped)} of {len(wanted)} identifiers along the route were "
            f"not asked about, to stay inside the RapidAPI allowance: "
            f"{', '.join(skipped[:8])}{'...' if len(skipped) > 8 else ''}"
        )

    return RouteNotams(
        notams=tuple(merged.values()),
        designators=asked,
        failed=tuple(dict.fromkeys(failed)),
    )


def _designators(
    positions: tuple[LatLon, ...], corridor_nm: float, priority: tuple[str, ...]
) -> tuple[str, ...]:
    """Every identifier to ask about, in the order the budget should keep them.

    First the route's own aerodromes, then the centres, then every other
    aerodrome nearest the track first. That order is the whole point: the cap
    takes from the front, so what falls off the end is the field farthest
    from the route -- not, as walking from departure to destination would
    have it, everything at the far end including the destination itself.
    """
    from engine import airports as apt

    radius = max(corridor_nm, 1.0) / _CHAIN_COVERAGE
    samples = tuple(point for point, _ in _query_points(positions, radius))
    try:
        found = apt.designators_along_route(samples, radius)
    except apt.AirportDatabaseMissing as exc:
        raise NotamsUnavailable(
            "SkyLink is asked by airport identifier, and the airport database "
            "that finds them is not built -- run `make airports`"
        ) from exc

    route_fields = tuple(dict.fromkeys(i.strip().upper() for i in priority if i))

    def off_track(ident: str) -> float:
        airport = apt.find(ident)
        if airport is None:
            return math.inf
        return min(inverse(sample, airport.position).distance_nm for sample in samples)

    others = sorted(
        (ident for ident in found.airports if ident not in route_fields),
        key=off_track,
    )
    return tuple(dict.fromkeys((*route_fields, *found.artccs, *others)))


def _query_points(
    positions: tuple[LatLon, ...], radius_nm: float
) -> tuple[tuple[LatLon, float], ...]:
    """A chain of circles covering the route, at most `MAX_QUERY_POINTS`.

    Every waypoint is sampled, since that is where the route's own aerodromes
    are, and long legs are filled in between at the spacing the radius
    supports. Where the route is too long for the cap, the spacing is
    stretched to fit and the corridor between the circles narrows.
    """
    spacing = max(radius_nm, 1.0)
    spans = [inverse(a, b) for a, b in zip(positions, positions[1:])]
    total = sum(span.distance_nm for span in spans)
    needed = len(positions) + int(total // spacing)
    if needed > MAX_QUERY_POINTS:
        # Spread what is allowed over the whole route rather than covering the
        # first half properly and the second half not at all.
        spacing = max(spacing, total / max(1, MAX_QUERY_POINTS - len(positions)))

    points: list[LatLon] = [positions[0]]
    for span in spans:
        steps = int(span.distance_nm // spacing)
        points.extend(span.point_at_nm(spacing * (n + 1)) for n in range(steps))
        points.append(span.end)

    return tuple((point, radius_nm) for point in _thinned(points, spacing))


def _thinned(points: list[LatLon], spacing_nm: float) -> list[LatLon]:
    """Drop points that sit almost on top of one another.

    A waypoint at the end of a leg lands next to the fill point just before
    it, and two circles a mile apart search the same ground twice.

    A point is only dropped when dropping it leaves no gap: what follows it
    must still land within one spacing of the circle before it. Thinning
    without that check is how a chain that looks right develops a hole in the
    middle -- drop the point at 46 nm and the one at 23 nm has to reach 50,
    which at this spacing it cannot. A point dropped for being *inside* the
    previous circle costs nothing, since that circle already searches it.
    """
    if not points:
        return []
    kept: list[LatLon] = [points[0]]
    for index, point in enumerate(points[1:], start=1):
        if inverse(kept[-1], point).distance_nm >= spacing_nm * 0.5:
            kept.append(point)
            continue
        following = points[index + 1] if index + 1 < len(points) else None
        if (
            following is not None
            and inverse(kept[-1], following).distance_nm > spacing_nm
        ):
            kept.append(point)
    return kept


def _cache_path(url: str) -> Path:
    from hashlib import sha256

    return CACHE / f"{sha256(url.encode('utf-8')).hexdigest()[:16]}.json"


def _get_json(url: str, *, refresh: bool = False) -> Any:
    """GET and decode, serving from the disk cache while it is fresh.

    The same shape as `wx_surface._get_json`, and separate from it because the
    credentials, the TTL and the failure consequences are all different: a
    weather source that will not answer leaves a plan on ISA, and a NOTAM
    source that will not answer leaves a pilot believing there are none.
    """
    path = _cache_path(url)
    if not refresh and path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                cached = json.load(handle)
            if time.time() - float(cached["fetched_at"]) < TTL_S:
                return cached["payload"]
        except (OSError, ValueError, KeyError):
            pass  # A corrupt cache entry is simply a miss.

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "X-RapidAPI-Key": os.environ.get(RAPIDAPI_KEY_ENV, ""),
            "X-RapidAPI-Host": SKYLINK_HOST,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # 401 and 403 are the key, and saying so beats "could not reach": the
        # fix is a subscription, not a network.
        if exc.code in (401, 403):
            raise NotamsUnavailable(
                f"SkyLink rejected the key ({exc.code}); check {RAPIDAPI_KEY_ENV} "
                f"and that it is subscribed to SkyLink on RapidAPI"
            ) from exc
        # 429 is the allowance, not the service, and the fix is different:
        # wait for the month to roll over, or pay for more.
        if exc.code == 429:
            raise NotamsUnavailable(
                "SkyLink rate limit reached (429) -- the monthly or per-minute "
                "allowance is used up"
            ) from exc
        raise NotamsUnavailable(f"SkyLink answered {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NotamsUnavailable(f"could not reach SkyLink: {exc}") from exc

    try:
        payload = json.loads(body) if body.strip() else {}
    except ValueError as exc:
        raise NotamsUnavailable("SkyLink returned malformed JSON") from exc

    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"fetched_at": time.time(), "payload": payload}, handle)
        tmp.replace(path)
    except OSError:
        pass  # A cache we cannot write is a slower program, not a broken one.

    return payload
