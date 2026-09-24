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

Live surface weather is the one thing that needs a connection. When it is available, the
temperature and altimeter setting at each airport come from the current METAR — or, for a
departure hours away, from the TAF and a forecast model — so takeoff and landing distances are
computed on the air that will actually be there. Without a connection you type them in, as
before, and everything else works unchanged.

NOTAMs need one extra thing: a SkyLink licence (1,000 queries). Without it every other
feature works and **Get NOTAMs** says what is missing rather
than reporting none — "no NOTAMs" and "no NOTAM service" look identical on a briefing and mean
opposite things.

```bash
export RAPID_API_KEY=...
make serve
```

SkyLink is asked by airport identifier, so a briefing costs one request per aerodrome and ARTCC
near the route, capped at 40 — about 25 briefings a month in busy airspace. The route's own
airports and the ARTCCs are always asked first, so the cap only ever drops outlying fields.

What comes back is filtered to the flight rather than dumped: within 10 nm of track, at an
altitude the aeroplane is actually at over that stretch, and in force while it is there. The
panel says how many of how many survived, so you can see the filter working.

```bash
make altimeter-trend   # how far the altimeter setting really moves in a day, and what it costs
```

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

Built, with 672 tests:

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
- [engine/csv_download.py](engine/csv_download.py) — the finished log as CSV in the
  column order of a printed nav log pad, for pasting into one
- [engine/weather.py](engine/weather.py) — METAR, TAF and model forecasts decoded and
  reconciled into one report per field, with the source recorded on every value. Pure and
  offline; [server/wx_surface.py](server/wx_surface.py) does the fetching
- [engine/tiff_reader.py](engine/tiff_reader.py) + [engine/charts.py](engine/charts.py) +
  [engine/chart_render.py](engine/chart_render.py) — FAA sectional and terminal area raster
  charts, read straight out of the GeoTIFF (no GDAL), reprojected from Lambert Conformal
  Conic into map tiles, and dated from the FAA's own metadata so the currency list says when
  each edition lapses
- [ui/](ui/) + [server/](server/) — MapLibre route editor with a live navigation log,
  on a 535 KB offline basemap, with the raster charts under it on a layer switch

## Charts

Sectional and terminal area charts are not in the repo — they are 30–80 MB apiece and the FAA
reissues them every 56 days. Download the ones you fly, unzip them into
`data/charts/sectional/` or `data/charts/tac/`, and they appear in the map's **Layers** button.
See [data/charts/README.txt](data/charts/README.txt).

```bash
make charts     # render them into map tiles, and commit those
```

The tiles *are* committed, unlike the charts. A deployed instance has no GeoTIFFs on it and
not enough memory to decode one, so pre-rendered tiles are the only form in which a chart
reaches a browser anywhere but this desktop. While developing you can skip the step: the dev
server renders each tile the first time the map asks for it.

## Sources
- FAA NASR dataset:  https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/
- FAA VFR raster charts: https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/vfr/

The NASR dataset requires updating every 28 days; VFR charts every 56.

Next: the FAA NASR pipeline, then the map UI, then corridor-constrained A\* routing. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §6 for the design of everything not yet written.
