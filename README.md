# my_e6b

Offline VFR cross-country flight planner for the Cessna 172S.

Enter departure and destination; get a suggested route with VFR waypoints, a cruise altitude,
and alternates. Edit the route by hand, then generate a navlog. Designed to run fully offline —
on a Linux desktop now, on an iPad as an installable PWA later.

> **Not for navigation.** A planning aid only. The performance data is digitized from an
> unofficial source and is unverified — see [data/poh/c172s/SOURCE.md](data/poh/c172s/SOURCE.md).
> Verify every number against a real POH and current charts before any flight.

## Getting started

```bash
make data       # build the airport database and offline basemap (once)
make serve      # then open http://127.0.0.1:8137
```

Type an identifier to add airports, click the map to insert a waypoint, drag markers to move
them, and the navigation log updates as you go. It works with the network off.

## Where things are

| Path | What |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **Start here** — design, rationale, and what is built vs. planned |
| [engine/](engine/) | The planning engine. Pure library: no UI, no network, no server imports |
| [data/poh/c172s/](data/poh/c172s/) | POH Section 5 tables and their provenance |
| [ui/](ui/) | The map and navlog. Static files, identical in dev and in the packaged app |
| [server/](server/) | FastAPI dev shim. Never shipped — the PWA calls the engine directly |
| [tools/](tools/) | Desktop-only build and verification scripts |
| [tests/](tests/) | Test suite |

## Status

Built, with 332 tests:

- [engine/atmosphere.py](engine/atmosphere.py) — ISA, pressure and density altitude, TAS
- [engine/performance.py](engine/performance.py) — POH Section 5 tables and interpolation
- [engine/geo.py](engine/geo.py) — spherical geodesics and the wind triangle, with polar
  latitudes refused
- [engine/magnetic.py](engine/magnetic.py) — WMM2025 variation, validated against NOAA's own
  100-point reference set
- [engine/navlog.py](engine/navlog.py) — full navigation log with top-of-climb and
  top-of-descent, fuel accounting and VFR reserve checking
- [engine/airports.py](engine/airports.py) — lookup over 15,762 US airports
- [engine/weight_balance.py](engine/weight_balance.py) — gross weight and centre of
  gravity from a loading form
- [ui/](ui/) + [server/](server/) — MapLibre route editor with a live navigation log,
  on a 535 KB offline basemap

## Sources
- FAA NASR dataset:  https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/

The NASR dataset requires updating every 28 days

Next: the FAA NASR pipeline, then the map UI, then corridor-constrained A\* routing. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §6 for the design of everything not yet written.
