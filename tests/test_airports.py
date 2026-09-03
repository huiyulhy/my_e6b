"""Airport lookup, and the one thing about it that is not obvious: threads.

The database is a committed build artefact, so these run offline like the rest
of the suite.

The concurrency tests are the point of this file. `engine/airports.py` is read
only and looks stateless, which is exactly why a shared connection survived
review: nothing about a lookup suggests it could interfere with another one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from engine import airports as apt

IDENTS = ("KSQL", "KLVK", "KMOD", "KSFO", "KSJC", "KPAO", "CN44")


class TestFind:
    def test_an_icao_identifier_resolves(self):
        airport = apt.find("KSQL")
        assert airport is not None
        assert airport.ident == "KSQL"
        assert airport.position.lat == pytest.approx(37.51, abs=0.05)

    def test_lookup_is_case_and_space_insensitive(self):
        assert apt.find("  ksql ").ident == "KSQL"

    def test_an_unknown_identifier_is_none_rather_than_an_error(self):
        assert apt.find("ZZZZ") is None

    def test_a_vfr_waypoint_is_not_an_airport(self):
        # VFR checkpoints live in their own table. `find` must not reach them,
        # or the weather panel would ask a station that does not exist.
        assert apt.find("VPMIN") is None


class TestPatternAltitude:
    """The height an overcast has to clear, where NASR publishes one.

    Only a few hundred US fields file a pattern altitude, so the absence of
    one is the normal case and must not read as a low pattern -- the go/no-go
    takes the standard 1,000 ft instead.
    """

    def test_a_published_pattern_comes_back_above_the_field_not_above_the_sea(self):
        # NASR's TPA is a height above the field. KSQL is 5 ft up and flies an
        # 800 ft pattern; an MSL reading would be 805.
        airport = apt.find("KSQL")
        assert airport.pattern_altitude_agl_ft == pytest.approx(800.0)
        assert airport.elevation_ft < 100.0

    def test_a_field_that_publishes_none_says_none(self):
        assert apt.find("KLVK").pattern_altitude_agl_ft is None


class TestConcurrentLookups:
    """One connection per thread, not one shared between them.

    A `sqlite3.Connection` is not safe for simultaneous use. Shared across
    threads it interleaves results on its cursor, which surfaced as
    `InterfaceError: bad parameter or other API misuse` -- and, worse, as rows
    carrying another query's columns. The weather panel resolves every field on
    the route at once, so these calls genuinely overlap.
    """

    def _hammer(self, work, count=400, workers=16):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(work, range(count)))

    def test_parallel_lookups_do_not_raise(self):
        def work(n):
            try:
                apt.find(IDENTS[n % len(IDENTS)])
            except Exception as exc:  # noqa: BLE001 -- the failure is the point
                return f"{type(exc).__name__}: {exc}"
            return None

        failures = [r for r in self._hammer(work) if r]
        assert not failures, f"{len(failures)} concurrent failures: {set(failures)}"

    def test_parallel_lookups_return_the_row_that_was_asked_for(self):
        """The quiet half of the bug.

        Errors are survivable -- a wrong row is not. A lookup that came back
        with another airport's position would put it silently into a flight
        plan, so this asserts identity rather than merely absence of an
        exception.
        """

        def work(n):
            ident = IDENTS[n % len(IDENTS)]
            try:
                airport = apt.find(ident)
            except Exception as exc:  # noqa: BLE001 -- reported, not raised
                return f"{ident}: {type(exc).__name__}"
            if airport is None:
                return f"{ident} not found"
            if ident not in (airport.ident, airport.icao, airport.iata):
                return f"asked {ident}, got {airport.ident}"
            # A row assembled from two queries loses its position first.
            if airport.position.lat is None or airport.position.lon is None:
                return f"{ident} came back with no position"
            return None

        wrong = [r for r in self._hammer(work) if r]
        assert not wrong, f"{len(wrong)} wrong rows: {set(wrong)}"

    def test_mixed_query_shapes_in_parallel(self):
        """`find`, `search` and `near` share the connection and differ in shape.

        The original corruption needed two queries returning different columns
        at once, which one repeated lookup never produces.
        """

        def work(n):
            try:
                match n % 3:
                    case 0:
                        apt.find(IDENTS[n % len(IDENTS)])
                    case 1:
                        apt.search("San", limit=5)
                    case _:
                        apt.near(apt.find("KSQL").position, 40.0, limit=10)
            except Exception as exc:  # noqa: BLE001 -- the failure is the point
                return f"{type(exc).__name__}: {exc}"
            return None

        failures = [r for r in self._hammer(work, count=300) if r]
        assert not failures, f"{len(failures)} concurrent failures: {set(failures)}"
