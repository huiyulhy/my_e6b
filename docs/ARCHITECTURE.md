# System Architecture

Offline VFR cross-country flight planner for the Cessna 172S. Enter departure and destination;
get a suggested route with VFR waypoints, a cruise altitude, and alternates; edit the route by
hand; generate a navlog.

**This is a planning aid, not a certified navigation system. Not for navigation.**

Status legend: **[built]** exists and is tested · **[next]** designed, not written ·
**[later]** shape decided, details open.

---

## 1. The two constraints that shaped everything

Every other decision follows from these.

**You develop on Linux.** Producing an iOS `.ipa` requires macOS and Xcode — for Qt, Flutter,
Tauri mobile, and Kivy alike — plus a $99/yr Apple Developer account to keep a sideloaded build
installed past 7 days. An installable PWA is the only route from a Linux desktop to an iPad
with no Mac and no fee.

**Pyodide runs CPython 3.13 in the browser.** Verified against the Pyodide 0.28 lockfile:
`numpy 2.2.5`, `scipy 1.14.1`, `shapely 2.0.7`, `networkx 3.4.2`, `pandas 2.3.0` all ship in
the distribution. So a Python engine runs *unmodified* offline on the iPad.

Together these mean Python is simultaneously the fastest path to a prototype and the only
free path to the iPad — normally a trade-off, here not one. The venv is pinned to **3.13** to
match Pyodide's CPython exactly.

**Absent from Pyodide, already planned around:** `pyproj`, `geopandas`, `rtree`.
Substitutes — geodesics: hand-written spherical formulae in stdlib `math`, so nothing has to
be vendored at all (§4a); spatial index: `scipy.spatial.cKDTree`; vector I/O: `json` +
`shapely`.

---

## 2. Layout

```
my_e6b/
  engine/          pure library: no I/O beyond its own data, no UI, no network
    atmosphere.py    [built]  ISA / pressure alt / density alt / TAS
    performance.py   [built]  POH Section 5 tables + interpolation
    geo.py           [built]  spherical geodesics + wind triangle
    magnetic.py      [built]  WMM2025 magnetic variation
    navlog.py        [built]  legs -> navlog rows
    airspace.py      [later]  shapely polygon queries
    corridor.py      [later]  candidate-node generation
    router.py        [later]  A* over the corridor graph
    weights.py       [later]  every tunable cost weight, one dataclass
    altitude.py      [later]  cruise altitude optimiser
    alternates.py    [later]  alternate ranking
    airports.py      [built]  airport lookup against the local SQLite
    weather.py       [built]  METAR/TAF/model parsing + source resolution (pure)
  data/
    poh/c172s/       [built]  five CSVs + SOURCE.md provenance
    magnetic/        [built]  WMM2025.COF + NOAA's 100-point test set
    aero/            [built]  airports.sqlite (OurAirports, a stopgap)
    basemap/         [built]  535 KB of Natural Earth GeoJSON
  tools/             desktop-only build and verification scripts
    validate_poh.py  [built]  dump every POH cell + consistency checks
    build_airports.py[built]  OurAirports -> SQLite
    build_basemap.py [built]  Natural Earth -> clipped, simplified GeoJSON
    screenshot_ui.py [built]  drive the map in a real browser
    plot_altimeter_trend.py
                     [built]  24 h of altimeter settings, and what they cost in feet
    build_nasr.py    [later]  FAA NASR / DOF / airspace -> SQLite
  ui/                [built]  static HTML/JS + MapLibre — same files dev and prod
  server/            [built]  FastAPI dev shim, never shipped
    main.py          [built]  the HTTP surface
    wx_surface.py    [built]  the only network fetch: AWC METAR/TAF + model (see 4d)
    wx_aloft.py      [later]  winds aloft at leg midpoints
  tests/             [built]  672 tests
  Makefile           [built]  every recipe clears PYTHONPATH (see §7)
```

### The engine boundary is the load-bearing rule

`engine/` imports nothing from `server/`, `ui/`, or `tools/`, touches no network, and reads
only its own bundled data. This is what lets one codebase serve both run modes, and it is the
constraint to preserve when modifying anything.

**Two run modes, one engine:**

| | Dev (now) | Ship (step 7) |
|---|---|---|
| Host | FastAPI on Linux desktop | Static bundle, no server |
| Engine | imported directly | same code under Pyodide |
| UI | `ui/` served over HTTP | identical `ui/` files |
| Data | `data/` on disk | baked into the bundle |
| Network | localhost | none — airplane mode |

The UI calls one narrow interface — `plan_route`, `edit_route`, `build_navlog` — that is either
an HTTP endpoint or a Pyodide call. Keep that boundary strict and the modes cannot drift.

**If Python is ever too slow**, only `engine/` gets ported to C++/WASM. `ui/` is never rewritten.
That is the entire reason the UI is a browser app rather than PySide6.

---

## 3. `engine/atmosphere.py` — [built]

ISA model - POH and altimeters are calibrated to this. **Stdlib `math` only**, 

**The model is capped at the tropopause** (36089 ft) and raises `AboveModelCeiling` above it,
rather than returning isothermal-layer numbers nobody has checked. A 172S service ceiling is
around 14000 ft and class A  is 18000 ft so we have no business at this altitude

Aviation-native units at the boundary (feet, Celsius, inHg, knots), SI internally.

Every quantity has an exact form and, where pilots use one, the cockpit rule of thumb beside
it — `pressure_altitude` vs `pressure_altitude_approx`, `density_altitude` vs
`density_altitude_approx`, `tas_from_cas` vs `tas_from_cas_approx`. Tests assert they agree to
a stated tolerance, which documents how much the shortcuts actually cost. (At 1000 ft / 29.42,
the 1000-ft-per-inHg rule is off by ~38 ft.)

`tas_from_cas` uses the full subsonic compressible relation. At C172 speeds it differs from
`CAS/√σ` by well under a knot, but it costs nothing and removes a caveat from the navlog.

---

## 4. `engine/performance.py` — [built]

POH Section 5 tables and interpolation. **Two rules govern this module:**

1. **Never extrapolate.** Outside the published envelope, raise `OutsidePOHEnvelope`. A planner
   that invents a takeoff distance for a 9000 ft pressure altitude is worse than one that
   refuses.
2. **Interpolate** Linear between published rows

### The tables are not uniformly shaped, and that drives the code

| Table | Shape | Wrinkle |
|---|---|---|
| takeoff | weight × press alt × temp | Full 3-D grid. Temps live in the **column names** (`groundroll_0`…`_40`), so rows are melted before gridding. |
| landing | press alt × temp | 2550 lb only — the POH publishes max weight only. Using it at lower weights is conservative, which errs the right way. |
| climb rate | press alt × temp | **One blank cell** at 12000 ft / 40 °C. |
| climb dist | press alt | Time and fuel are cumulative from sea level — a segment is the difference of two rows. The speed column is read directly and averaged over the segment; the distance column is not read at all (see below). |
| cruise | (alt, RPM) × temp | **Ragged**: 2100 RPM exists only at 2000–4000 ft, 2700 RPM only at 8000–10000 ft. Temp is always full. |

### Two mechanisms handle the irregularity

**Chart holes — the `_Grid` mask.** A blank cell cannot simply sit in the value array as NaN,
because NaN spreads to every neighbouring query, including one landing exactly on a *published*
corner — `0 * NaN` is still NaN. That would refuse climb rate at 10000 ft / 20 °C, which the
POH does publish. So `_Grid` carries a **parallel validity interpolator**: values interpolate
with the hole zeroed, and the mask reports how much of each query's weight came from published
cells. Answerable only if all of it did.

**Ragged cruise — Delaunay, not a regular grid.** A `RegularGridInterpolator` needs a full
grid; cruise has none. Instead the `(altitude, RPM)` points that actually exist are triangulated
once — **axis-normalised first**, since altitude spans thousands and RPM hundreds, and raw
Delaunay would be badly conditioned. One `LinearNDInterpolator` per published temperature shares
that triangulation; results blend linearly in temperature. Queries landing in a hole return NaN
and become a refusal. Falling outside the convex hull is refused for free — the desired
behaviour, obtained structurally rather than by a bounds check.

`available_cruise_rpm(alt, temp)` enumerates settings that actually exist at an altitude, so
the altitude optimiser can iterate real options instead of guessing and being refused.

### Chart notes implemented — all need verification against your POH

Wind and surface corrections (10% per 9 kt headwind, 10% per 2 kt tailwind, +15% of ground roll
for takeoff on dry grass, +45% for landing) and the climb temperature note (±10% per 10 °C from
standard, applied to time and fuel at the segment midpoint, in both directions so a cold day
is credited — floored at half the published figure so no forecast climbs in no time).

---

## 4a. `engine/geo.py` and `engine/magnetic.py` — [built]

**Geodesy is spherical**, hand-implemented in stdlib `math`. Not `geographiclib`, which is
*absent from Pyodide* and would have to be vendored as a wheel and loaded through `micropip`
at page start.

The ellipsoid was tried first and measured against, then dropped as unnecessary:

| leg | distance error | bearing error |
|---|---|---|
| KSQL–KPAO 7 nm | −0.01 nm | 0.09° |
| KSQL–KMRY 59 nm | +0.09 nm | 0.08° |
| KSQL–KSAN 379 nm | +0.13 nm | 0.13° |


The `vincenty_inverse` reference implementation that produced the table has since been removed
along with the tests that used it, so **the figures above are a one-time measurement, not a
maintained bound** — nothing now re-checks them if the geodesics change. Restore an ellipsoidal
reference if that guarantee is wanted back.

**Two invariants matter more than absolute accuracy, and both come from using one model
everywhere:**

- `direct` and `inverse` must be exact inverses (they round-trip to 2e-13 nm). The navlog
  places TOC and TOD by flying a bearing for a distance and then re-measures the legs; if
  placement and measurement disagreed, distances would stop summing to the route length.
- The router's A\* heuristic must never *overestimate*. Spherical distance can come out larger
  than ellipsoidal, so a spherical heuristic against ellipsoidally-measured edges would silently
  break admissibility. One model removes the trap.

**Polar latitudes are refused** above `MAX_LATITUDE_DEG = 80`, raising `PolarRegionUnsupported`.
Near the poles meridians converge so sharply that a single "true course" stops describing the
path, and the sphere's disagreement with the real earth is at its worst. `direct` checks the
*resulting* point too, not just the departure.


**Magnetic variation is a full WMM2025 implementation**, degree-12 spherical harmonics, again
stdlib-only. Correctness here is not argued from first principles: NOAA ships 100 reference
points alongside the coefficients, and `tests/test_magnetic.py` checks every one. Max
declination error is **0.005°**, which is exactly half a unit in NOAA's last published decimal
place — i.e. their rounding, not ours. Field components agree to 0.0007 nT.

The coefficients expire in 2030; queries outside the five-year window raise
`OutsideModelValidity`, with no extrapolation path. `build_navlog` checks the planned date once
up front, so an expired model fails the whole plan rather than silently producing wrong magnetic
courses. Refresh from NOAA when it lapses.

---

## 4b. `engine/navlog.py` — [built]

Where the other four modules meet. It owns no physics; it sequences theirs. Five modelling
decisions, all visible in the output:

- **User-driven planning is the default.** The pilot declares what each leg does and the
  altitudes follow from performance; an undeclared leg is refused rather than guessed at, and
  the cruise altitude is a bound rather than a target. `planning_mode="auto"` asks for the
  other behaviour, where the planner picks the profile from a stated cruise altitude.
- **A wind typed on a row is planned with, not just flown with.** It reaches the profile
  before the tops of climb and descent are placed, so a headwind on the climb row moves the
  TOC back down the route. It belongs to the leg, not to an altitude: it holds all the way up
  the climb, where a typed *temperature* is hung at one altitude and interpolated between.
  Automatic mode needs one rehearsal lay-out to learn which drawn leg a row sits inside, the
  same way row temperatures do.
- **TOC and TOD are spliced in as real waypoints.** No leg spans a phase change, so every row
  has one altitude, one TAS and one fuel flow. This is why the arithmetic stays simple and
  still correct.
- **Climb and descent ground distances come from ground speed, not the POH.** The POH climb
  table's *distance* column is still-air distance; its *time* and *fuel* columns are air-mass
  quantities that stay valid in wind. So time and fuel are used directly, and the distance is
  flown: the band's climb speed (the average of the table's speed column over the band) as TAS
  against the band's forecast wind, times the band's time.
- **The POH publishes no descent table.** Descent is a constant rate at a chosen airspeed, with
  fuel charged at the cruise rate — conservative, configurable, and stated in the output. That
  rate is read at the altitude the descent *begins* from, not at the nominal cruise altitude,
  which in user-driven mode is only a bound the aeroplane may never reach.

Winds aloft interpolate on **vector components, not direction**: halfway between 350° and 010°
is 000°, but the arithmetic mean is 180°. The engine still takes a `WindsAloft` profile and
`profile.py` uses it to place TOC and TOD, but **the app no longer feeds it one**: an FD level
is a single wind for a quarter of a state, and on a route that crosses a coast range it is
wrong on one side of the hills or the other. Wind is typed per navlog row instead, as a
`LegOverride`, where it belongs to the leg it was forecast for. A row with nothing typed on it
is calm, which also means the vertical profile is placed at zero wind and a wind typed on a row
does not move the top of climb.

Fuel accounting includes the POH taxi allowance and checks the FAR 91.151 reserve (30 min day,
45 min night at cruise burn). Warnings are data on the `Navlog`, not printed side effects, so
the UI can surface them however it likes.

---

## 4c. `ui/` and `server/` — [built]

A MapLibre map with a route editor and a navlog that updates as you type.
`make serve`, then <http://127.0.0.1:8137>.

**Nothing is fetched from the network at runtime.** That constraint drove three decisions:

- **The basemap is local GeoJSON, not tiles.** Natural Earth clipped to a US box and
  simplified — 998 KB for states, coastline, lakes and major highways, down from 53 MB of
  source. Roads are the bulk of both numbers: the world file is 56,600 features, and
  filtering to US major highways leaves 2,014, which is 463 KB instead of fifty megabytes.
  At VFR planning zoom the simplification is invisible, and no tile pipeline or server is
  needed.
- **MapLibre is vendored** into `ui/vendor/` (939 KB). A CDN link would defeat the point.
- **No `symbol` layers.** MapLibre's `text-field` requires a `glyphs` URL serving font PBFs —
  a network dependency. Airport identifiers are rendered as HTML markers instead, capped at 60
  and only above zoom 8, since they are real DOM nodes.

Two layers carry names that cannot be drawn without those glyphs, so they answer on hover
instead: a highway reports its designation (`I-980`), and Class B/C/D airspace reports every
shelf under the cursor with its floor and ceiling. Hover rather than click, because a click on
the map inserts a waypoint. `tools/build_airspace.py` keeps B, C and D from the NASR shapefile
and drops Class E — 8 of the 12 million source vertices are the E5 700 ft blanket, which as an
outline is a wash rather than information.

**The API boundary is three calls** — search airports, list airports in view, plan a route.
In `ui/app.js` they live in one `api` object at the top of the file; swapping it for Pyodide
calls is the entire desktop-to-iPad change. Keep planning logic out of the browser or the two
run modes will diverge.

Engine refusals are returned as `{ok: false, error}` rather than HTTP errors, so
"no published cruise data at 13500 ft" appears in the UI as an explanation instead of a stack
trace. The refusal philosophy reaches all the way to the screen.

### `make ui-shot` — because the map cannot be unit tested

MapLibre needs WebGL, and a bad layer style fails silently at *runtime*. `tools/screenshot_ui.py`
drives real Chrome through Playwright, builds a route, and reports console errors. Three real
bugs it caught that nothing else would have:

1. The `symbol` layer threw on every load (the glyphs problem above), so airport labels never
   appeared. MapLibre logs style validation failures rather than raising, so it was silent.
2. `flyTo` in the search handler ran *after* `addWaypoint`'s `fitBounds`, undoing it — the
   route was framed and then immediately unframed, leaving the departure off screen.
3. The route line was drawn by indexing `route` with the leg number. The plan has *more* legs
   than the route has waypoints, because TOC and TOD are spliced in, so every leg took the
   colour of the first phase — the whole line was climb-orange. Fixed by having legs carry
   their endpoint positions, which is also what lets TOC and TOD be marked on the map at all.

Headless Chrome has no GPU, so the script forces software WebGL. Without that MapLibre never
fires `load` and the map stays blank — which is exactly what a naive screenshot shows.

---

## 4d. `engine/weather.py` + `server/wx_surface.py` — [built]

Live surface weather for a field, so a go/no-go is computed on the air that is actually there
rather than on 29.92 and ISA. **The split is the point of this section**, because the obvious
split is the wrong one.

### Why it is not split by source

The natural instinct is one module per provider: AWC over here, the model over there. That
fails, because the sources do not divide along the question being asked:

| | wind | vis / ceiling | **temperature** | **altimeter** |
|---|---|---|---|---|
| METAR (observation, now only) | ✅ | ✅ | ✅ | ✅ |
| TAF (forecast, to 24-30 h) | ✅ | ✅ | **✗** | **✗** |
| model (any point, any hour) | ✅ | — | ✅ | ✅ |

**A TAF carries no temperature and no altimeter setting.** Not usually-missing — structurally
absent; the decoded `fcsts` blocks have `"altim": null` and `"temp": []`. So a TAF alone cannot
produce a pressure altitude, cannot produce a density altitude, and therefore cannot produce the
takeoff and landing distances it was fetched for. The model is needed on the *surface* tier, not
just aloft, and any split by provider cuts straight through the middle of one question.

The split that holds is **fetch vs decide**:

- `engine/weather.py` — pure. Types, parsers, and `resolve_surface`, the one place the choice
  between sources is made. Takes decoded payloads, opens no socket, and so keeps working under
  Pyodide and under test.
- `server/wx_surface.py` — all the I/O: three concurrent stdlib `urllib` GETs, a `.cache/wx/`
  disk cache with per-product TTLs, and the nearest-TAF fallback.

This is the engine boundary from §2 applied to a feature that is *entirely* about the network.
It survives the boundary by having its judgement separated from its plumbing.

### What `resolve_surface` decides

Per field, with the winning `Source` recorded on every one — a mixed result is the normal case,
not an exception:

- **Target is now and the METAR is current** → the METAR wins on everything it publishes.
  An observation beats a forecast of the same moment.
- **Target is in the future** → TAF for wind, gusts, visibility and ceiling; **model for
  temperature and altimeter**. This is what makes a target-time go/no-go possible at all.
- **The METAR is dropped entirely once the target is past it**, rather than kept as a last
  resort. Most fields here are part-time — KSQL's "latest" observation is three hours old at
  2 am — and its temperature is exactly the field that moves. An afternoon takeoff computed on
  the morning's OAT reads short on the hottest day of the year. The model is fetched even for a
  "now" request so there is something to fall back to.
- **No TAF at the field** → borrow the nearest one that covers the target (`airports.near`,
  already a database read), tagged `NEAREST_TAF` and named in `notes`. Only ~600 US airports
  issue a TAF and most GA fields are not among them, so this is the common path, and it is
  labelled rather than silent.
- **Anything nobody supplies stays `None`**, with a note saying distances cannot be computed.
  Same rule as `OutsidePOHEnvelope`: refuse rather than guess.

### Two facts that will bite whoever touches this next

- **AWC reports `altim` in hectopascals** — `1012.6`, while the `rawOb` beside it says `A2990`.
  Read as inches it is thirty-four times too large, and the wrong pressure altitude is confident
  and unsafe. `hpa_to_inhg` refuses anything outside 25–32.5 inHg rather than converting
  silently, and tests assert the decoded field against the raw text.
- **`visib` is a number on one observation and a string on the next** (`10`, `"10+"`, `"1 1/2"`).
  Both appear in a single 24-hour response.

Model pressure comes from `pressure_msl`, never `surface_pressure`: an altimeter setting is
sea-level-reduced by definition, and station pressure would double-count field elevation when
`pressure_altitude` applies it again.

### How it reaches the go/no-go

It needs no engine change. `preflight._check_one_runway` already reads only OAT and pressure
altitude, both derived from `field_weather()`, which already reads `Waypoint.altimeter_inhg` and
`Waypoint.oat_c`. **Get weather**, over the navlog, calls `GET /api/wx/surface?ident=` once per
airport on the route and fills the boxes the plan already sends: altimeter setting, temperature,
and now wind and gust. At KMOD that moved density altitude from 97 ft to 1052 ft and the takeoff
over a 50 ft obstacle from 1644 ft to 1744 ft.

Only what the report actually gives is written — a station with no temperature leaves that box
alone rather than blanking it — and a wind is written whole, direction, speed and gust together,
because half a wind is not one and a stale gust would be held against the crosswind limit after
it had dropped. Each field then carries a provenance line (`KSQL 1953Z — wind metar · OAT metar ·
alt set metar`) that disappears from any box the pilot has since typed over: a line still
claiming a METAR said something it did not is the failure this panel exists to prevent.

The surface wind reaches the distances too. `Waypoint.wind_from_deg` / `wind_speed_kt` /
`gust_kt` carry it as reported — **true**, the way a METAR gives it — and `navlog._field_wind`
converts it to magnetic through the same WMM the navlog's headings come from, because a runway
designator is magnetic and 13°E of variation on the west coast is a quarter of the way to the
next designator: enough to move a crosswind across the demonstrated limit.

`preflight` then works end by end. `Runway.ends` parses the designator (`09L/27R` → 90°, 270°),
`wind_components` resolves the wind onto each, and `_pick_end` takes the end with the better
headwind — the end a pilot would pick, and the only one whose distances mean anything. That
headwind goes to `perf.takeoff_distance`/`landing_distance`, whose POH correction was already
there and previously always called with zero.

Two asymmetries are deliberate:

- **Gusts count against you, never for you.** A gusting headwind is read at the steady wind (the
  gust may not be there on the roll); a gusting tailwind is read at the gust (it may well be);
  the crosswind is always read at the peak, because the gust is the part that runs out of rudder.
- **Crosswind and length are separate gates.** A 10,000 ft runway 90° across a 25 kt wind fails,
  and it fails *as a crosswind* — `_why_no_runway` names the gate that closed, because "wait for
  the wind" and "do not go" are different decisions. Beyond 10 kt of tailwind the chart's
  correction is refused outright rather than extrapolated: 10% per 2 kt compounds too fast to
  guess past the published end.

A field with no wind on it keeps the zero-wind book figures and says so in every runway's note.
Silently reading "no wind given" as "calm" would be the one failure mode worth avoiding here.

`tools/plot_altimeter_trend.py` is the evidence behind all of this — 24 h of settings for six
Bay Area fields in one request, plotted in inches and in the pressure altitude error they cause.
Over a representative day each field moved about 0.10 inHg (~95 ft), and two fields 60 nm apart
differed by 0.12 inHg (~112 ft) at the same moment. The spread across the area is the better
argument than the drift over time: it is why one field's ATIS should not be used for the next
airport down the route.

### Not built: winds aloft

The aloft tier is designed but unimplemented. When it lands it must request
`geopotential_height_{level}hPa` alongside the winds and interpolate on **actual height** — the
`5000 ft ≈ 850 hPa` table is the ISA mapping and isobaric surfaces move with the pressure field.
Build a `WindsAloft` ladder and let its existing vector interpolation do the work rather than
hand-rolling u/v. Feed it through `profile.resolve_route(leg_winds=)`, sampling every leg
midpoint in one batched request. Note that doing so revisits §4b's decision that a typed wind
does not move the top of climb: the reason given there was that an FD level is too coarse to
trust, and point-resolved model data is not.

---

## 5. Verification strategy

The performance data is **digitized from an unofficial source with no license and no accuracy
guarantee** (see `data/poh/c172s/SOURCE.md`). Everything the planner computes rests on it. So
verification is a first-class component, not an afterthought.

**`make test` — 332 tests.** The central one is `TestGridPointsAreExact`: interpolating at a
published grid point must return the published number for *every cell of every table*. This is
what catches reshaping errors — a transposed axis still produces plausible numbers, just not
the right ones.

**`make validate`** prints all 199 lines of published data in POH chart layout for page-by-page
diffing, then runs four internal-coherence checks.

### Two lessons already learned from these checks — both worth knowing before you modify them

**A test can be wrong about aerodynamics.** An early test asserted TAS rises with altitude at
fixed RPM. It does not: a normally-aspirated engine loses power as it climbs, so at 2400 RPM
power falls 61% → 51% and TAS falls 109 → 104 kt. TAS gains with altitude only at *constant
percent power*, which needs higher RPM up high — at a matched ~57%, TAS rises 104 → 112.5 kt on
a flat 8.2 gph. Both behaviours are now asserted separately.

**A check can be too strict.** The first fuel-flow check flagged five cells, all at the lowest
power settings. That was real physics — specific fuel consumption worsens at low power, so those
cells genuinely burn more gallons per percent. Replaced with ordering checks. Those then flagged
TAS at 2000 ft / 2550 RPM (117/118/117 across temperature), which is *also* real: at full
throttle down low, warmer air costs power but thins simultaneously, and the effects nearly
cancel. The chart shows the progression plainly — TAS spans 3 kt at 2100 RPM, 2 kt at 2400, is
exactly flat at 2500, and goes slightly non-monotonic at 2550. So power and fuel flow must fall
with temperature; **TAS is deliberately exempt**.

**And a third time, in the navlog.** A test asserted that a hot day burns more fuel. It burns
*less*. At a fixed 2400 RPM, ISA+15 drops cruise from 55% to 52% and fuel flow from 7.90 to
7.58 gph; climb fuel does rise, 2.80 → 3.22 gal, and the flight is slower — but the cruise
saving outweighs the climb penalty. A fixed RPM is simply a lower power setting when it is hot.

The general lesson: when a check fires, establish whether the *data* or the *check* is wrong
before changing either. Three of the four surprises so far have been in the test, not the code
— and every one of them was the same misconception, that fixed RPM means fixed power.

---

## 6. Design of the parts not yet built

### Data pipeline → SQLite — [later]

`tools/` parses FAA sources once per 28-day cycle into **one SQLite file** shipped as a build
asset. Language-neutral, so a later C++ port reads it unchanged.

| Source | Gives |
|---|---|
| NASR `APT.txt` | airports, runways, surface, lighting, pattern direction, fuel |
| NASR `FIX.txt` | RNAV fixes and **VFR Visual Reporting Points** — the magenta-flag checkpoints, the highest-value layer for route suggestion |
| NASR `NAV.txt` | VOR/VORTAC/NDB, frequencies, variation |
| NASR `PJA.txt` | parachute jump areas — a routing *hazard*, not airspace |
| FAA AIS ArcGIS | Class B/C/D/E + SUA polygons, GeoJSON/shapefile |
| FAA DOF (56-day) | every obstacle above 200 ft AGL |
| SRTM / Copernicus | terrain for MSA |

**Two corrections to note.** Airspace boundaries are *not* in the pipe-delimited NASR set, but
they do not need the AIXM 5.1 flavour or the ArcGIS portal either — the ordinary NASR
distribution ships them as ESRI shapefiles under `Additional_Data/Shape_Files/`, which is what
`tools/build_airspace.py` reads. And those are **vector**; GeoTIFF is the separate raster
sectional imagery (d-VC), optional and only for display.

**Spatial index:** prefer SQLite R\*Tree, but **verify early whether the Pyodide sqlite3 build
includes the R\*Tree extension** — it may not. Fallback is `cKDTree` built at load, which the
router wants anyway. Keep the query layer indifferent to which backs it.

### `corridor.py` + `router.py` + `weights.py` — [later]

Great-circle plus VFR landmarks, as a weighted graph search so constraints are tunable.

1. **Corridor** — nodes within `clamp(0.15·D, 8 nm, 25 nm)` of the great circle.
2. **Nodes** — VFR waypoints, airports, navaids, fixes, significant towns/lakes; `cKDTree`.
3. **Edges** — k-nearest (k≈8) plus anything within 25 nm, filtered to forward-progressing only
   (bearing dot-product with the track > 0), which keeps the graph small.
4. **Cost** — every term a named field in `weights.py`: distance, turn penalty, airspace
   penetration (B/C/D graded, Prohibited/Restricted effectively ∞), terrain clearance,
   glide-range-to-airport, landmark identifiability, headwind.
5. **Heuristic** — `h(n) = remaining_gc_nm · w_dist_min`.

> **The one subtle correctness requirement.** Landmark preference must be a **floored discount**
> on distance, never a negative cost. If a bonus can make an edge cheaper than
> `w_dist_min · length`, the heuristic stops being admissible and A\* stops returning optimal
> paths. There is a test for this invariant; do not remove it.

6. **Post-process** — drop near-collinear waypoints, enforce min leg ≈5 nm, max ≈40 nm.

### `altitude.py` + `alternates.py` — [later]

Altitude is a separate 1-D pass once the path is fixed: enumerate legal VFR cruising altitudes
(hemispheric odd/even +500 above 3000 AGL), reject below MSA+1000 or above the practical ceiling,
cost each via the climb and cruise tables plus winds, return the top 3 with time and fuel.

Alternates rank by distance from route, runway length ≥ 1.5× POH landing distance, surface,
lighting, fuel. Both destination alternates and en-route emergency options.

---

## 7. Working in this repo

```bash
make test       # 332 tests
make validate   # dump POH tables + consistency checks
make check      # lint + test
make serve      # FastAPI dev server (once server/ exists)
```

**Always go through the Makefile, or clear `PYTHONPATH` yourself.** Sourcing
`/opt/ros/humble/setup.bash` puts ROS 2's Python **3.10** site-packages on `PYTHONPATH` for every
shell. Those leak into this project's 3.13 venv, and pytest then autoloads ROS's `launch_testing`
plugin, which fails on import. The venv is fine — only the inherited `PYTHONPATH` is the problem.
Every Makefile recipe clears it.

Note also that `uv` was originally installed under `~/snap/code/250/.local/bin` — VS Code's snap
sandbox redirects `$HOME`, and that path dies on the next VS Code update. It now lives at
`~/.local/bin/uv`.

### Conventions worth preserving

- `engine/` stays import-clean: no server, no UI, no network.
- Refuse rather than guess. `OutsidePOHEnvelope` over a plausible number, everywhere.
- Aviation-native units at API boundaries; SI internally; unit in every field name
  (`ground_roll_ft`, `fuel_gal`, `time_min`).
- Frozen dataclasses for results — `GroundDistance`, `ClimbSegment`, `CruisePoint` — not tuples.
- Anything transcribed from a chart note carries a "verify against your POH" comment.
