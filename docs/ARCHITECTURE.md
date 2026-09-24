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
    aloft.py         [built]  pressure-level winds and temperatures aloft (pure)
    planwx.py        [built]  the weather list beside the navlog, settled against it
    notam.py         [built]  NOTAM decoding + the 4D relevance filter (pure)
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
    wx_surface.py    [built]  the only network fetch: AWC METAR/TAF + model,
                              surface and pressure-level (see 4d)
  tests/             [built]  750 tests
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
| cruise | alt → (RPM × temp) | **Ragged**: 2100 RPM exists only at 2000–4000 ft, 2700 RPM only at 8000–10000 ft. Temp is always full. Altitude is a page *selector*, not an axis — see below. |

### Two mechanisms handle the irregularity

**Chart holes — the `_Grid` mask.** A blank cell cannot simply sit in the value array as NaN,
because NaN spreads to every neighbouring query, including one landing exactly on a *published*
corner — `0 * NaN` is still NaN. That would refuse climb rate at 10000 ft / 20 °C, which the
POH does publish. So `_Grid` carries a **parallel validity interpolator**: values interpolate
with the hole zeroed, and the mask reports how much of each query's weight came from published
cells. Answerable only if all of it did.

**Ragged cruise — one page, selected by rounding the altitude down.** A
`RegularGridInterpolator` needs a full grid; cruise has none, because the RPM list slides upward
and narrows with height. So cruise is stored one altitude page at a time, and **altitude is a
chart selector rather than an interpolation axis**: a query is rounded *down* to the published
page at or below it, and that single page is read at temperature and then RPM. Nothing blends
between pages.

Rounding down rather than interpolating buys two things:

- **It errs safe on fuel.** The lower page is denser air and more power, so it reads a higher
  burn per nautical mile — 0.0789 against 0.0769 gal/nm at 7,900 ft read on the 6,000 ft page at
  2500 RPM, the worst case on this chart, about 2.6%.
- **It makes the raggedness answerable.** Blending two pages meant the *upper* page's omission
  could refuse a setting the pilot's own altitude publishes perfectly well — 2100 and 2550 RPM
  above 4,000 ft, 2200 above 8,000 ft were all refused between pages for no reason the POH
  gives. Reading one page asks only whether *that* page has the setting.

The refusals that remain are settings missing from the page at or below the query — 2650 below
6,000 ft, 2700 below 8,000 ft. Those are real limits on the power available that low, not holes
to interpolate across.

The price is paid on the other side of the same coin: **TAS is read optimistically**, 114 KTAS
where 7,900 ft would give 112.1, so legs plan about a minute per 100 nm quicker than they fly.
Fuel per nautical mile still errs safe, but ETAs off this chart run slightly early and anything
reading ground speed inherits that.

`available_cruise_rpm(alt, temp)` enumerates settings that actually exist at an altitude, so
the altitude optimiser can iterate real options instead of guessing and being refused. Between
pages it reports the lower page's list, since that is the only page a query there reads.

### What counts as an extrapolation, and what happens to it — `perf.OffChart`

The rule is one sentence: a query past the edge of a chart is **refused**, or answered from a
**published cell beside it** — never from a curve fitted past the last row. `_bracket` raises
outside an axis, `_Grid`'s mask refuses a query drawing on a blank cell, the selected cruise
page refuses an RPM it does not publish, and a tailwind past the 10 kt the correction covers is
refused rather than projected down a slope that compounds at 10% per 2 kt.

But refusing *everything* off the edge fails ordinary days for ordinary reasons. A high-pressure
morning at a sea-level field is below the bottom row of every takeoff chart; ISA+25 over
California in July is off the right of the cruise chart's temperature band. Both have a
published cell that can honestly stand in, so one is read — and `OffChart` records that it
happened, with the field that decides what to do about it:

- **`conservative=True`** — the substitute errs safe. Below the bottom row the air is denser
  than the chart admits, so the distance reads long and the climb reads flat. A plan built on it
  is still a plan you can fly.
- **`conservative=False`** — the substitute errs the other way, or both ways. The equal-density
  cruise reading is accurate to about 1.6% on fuel flow and TAS *in either direction* — it was
  about 1% while altitude was interpolated, and rounding the substituted reading down to a page
  compounds the two disagreements; a level
  stretch below the chart's 2000 ft floor is read where the engine makes less power than it
  really will, so the fuel flow reads low and the reserve looks better than it is.

**The cruise band is the rule: ISA ±20 at the queried pressure altitude.** The chart prints
three columns per altitude page — ISA−20, ISA, ISA+20 — and a query inside that band is the
chart's own reading. Outside it there is no column for the operating point, and
`cruise_at_density` answers from air of the same *density* found elsewhere on the chart, at a
different pressure altitude and a different temperature. However carefully that is done, the POH
does not publish this aeroplane's cruise performance at the altitude and temperature asked for,
and the number standing in for it was measured somewhere else — so it is an **extrapolation**
and is labelled one. `cruise_isa_band()` reads the band off the digitized data rather than a
constant, so a re-digitized chart widens it instead of disagreeing silently.
`CRUISE_ISA_TOLERANCE_C` is **0**: the line sits exactly at the chart's edge, so ISA+21 counts
and ISA+20 does not. Past the point where no page covers the density at all, the answer is a
refusal rather than an extrapolation.

The record travels: `GroundDistance.off_chart` → `RunwayCheck` → `AirportCheck` (deduped, since
a field's pressure altitude produces the identical record on every runway) → `GoNoGo`. The
cruise side travels `CruiseLookup.off_chart` → the navlog's per-row collector → `Leg.off_chart`
→ the same `GoNoGo`.

**Three verdicts, not two.** `is_go` is unchanged — blockers and unknowns only. Letting an
off-chart reading force `NO GO` would put it on a high-pressure morning at Palo Alto, which is
both wrong and the fastest way to teach a pilot to ignore the word. `verdict` carries the third
state instead: `GO`, `GO -- EXTRAPOLATED`, `NO GO`. Conservative entries are listed first so
the list ends on the ones worth stopping for.

**The first row is the one that matters.** Time and fuel are cumulative, so a substitution on
row 3 is in every total from row 3 on whether or not rows 4 and 5 were read off-chart
themselves. `Navlog.first_extrapolated_leg` is that row; the printed log marks affected rows
with `+` and says where the totals stop being the book's, and the checklist groups repeats
behind the first occurrence rather than repeating one sentence per row.

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
is 000°, but the arithmetic mean is 180°. `profile.py` uses a `WindsAloft` profile to place TOC
and TOD, and for a long time the app fed it none: an FD level is a single wind for a quarter of a
state, and on a route that crosses a coast range it is wrong on one side of the hills or the
other. Wind was typed per navlog row instead, as a `LegOverride`, where it belongs to the leg it
was forecast for — and a row with nothing typed on it was calm, so the vertical profile was placed
at zero wind and a typed wind did not move the top of climb.

**Get weather** now fetches a point-resolved model column per leg, which is the objection
answered rather than argued with, so a forecast wind *does* move the tops of climb. See §4d, "How
it reaches the navlog". A typed wind still wins on the row it was typed against, and a route with
no forecast on it behaves exactly as described above.

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
- **The gust is not resolved on its own — it belongs to the wind.** Per-field resolution is right
  for every other field and wrong for this one: a METAR that gives a wind and omits the gust group
  is *stating* there is no gust, not leaving a hole. KHAF reporting `00000KT` under a model
  forecasting 9 kt resolved to "calm, gusting 9" — a peak that appeared in neither source, which
  the go/no-go then resolved onto a runway as crosswind. A gust whose source is weaker than the
  wind it would attach to is dropped, and the drop is noted. Same-source gusts (an observed
  `18015G25KT`, a TAF block with its own `wgst`) are untouched.

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

### Each field is fetched for the time you reach it, not the time you click

**Off blocks**, beside the button, is the departure time in Zulu. Left blank — the common case —
every field is fetched for now. Set, the fetch becomes a two-pass operation:

1. Plan, to get the ETEs. `plannedArrivals()` walks `lastPlan.legs`, takes the
   `cumulative_ete_min` of the first leg *arriving* at each airport, and adds it to the
   off-blocks time. The departure field is the off-blocks time itself.
2. Fetch each airport at its own ETA, then re-plan on what came back.

The alternative — one time for the whole route — is wrong in exactly the case the feature exists
for. A three-hour leg landing at 1600Z planned on the 1300Z temperature reads the destination
several degrees cool, and it is the afternoon arrival at a hot high field where the runway margin
actually gets thin. The ETEs are already computed and already on the log; using them costs one
extra plan and removes a whole class of quiet error.

Arrival rather than departure time at an intermediate stop: the runway check there is about the
landing, and the pattern and taxi that follow are ten minutes against weather published by the
hour. The provenance line carries the time it used (`KLVK 1116Z — wind taf · OAT model`), so a
forecast never reads as an observation.

The time is read as **UTC, not browser-local**. `datetime-local` carries no timezone, the label
says Z, and a pilot planning a Zulu departure should not have the machine's timezone silently
move it.

Only what the report actually gives is written — a station with no temperature leaves that box
alone rather than blanking it — and a wind is written whole, direction, speed and gust together,
because half a wind is not one and a stale gust would be held against the crosswind limit after
it had dropped. Whether the gust that arrives belongs to that wind is `resolve_surface`'s
decision, not the UI's; see above. Each field then carries a provenance line (`KSQL 1953Z — wind
metar · OAT metar · alt set metar`) that disappears from any box the pilot has since typed over:
a line still claiming a METAR said something it did not is the failure this panel exists to
prevent.

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

### The sky is a gate of its own

The same report carries the cloud and the visibility, and `preflight.check_weather` decides
whether the field is VFR. It is a second gate beside the runways, on the same footing as
crosswind is beside length: the longest runway on the field is no use under an obscuration, and a
clear sky does not lengthen a short one. `AirportCheck.runways_pass` keeps the old runway-only
verdict; `AirportCheck.passes` is the two of them together, and `summarise` reports each as its
own blocker because "no runway is long enough" and "the field is under an overcast" are different
problems with different answers.

Three rules, in the order they bite:

- **A vertical visibility is not a ceiling.** `VV` in a TAF, `OVX` in a METAR: there is no cloud
  base to stay under and no horizon, so the height reported is how far up you can *see*, not how
  far up you can fly. It ends the check and is never compared against a pattern altitude.
- **An overcast has to clear the pattern by 500 ft.** Basic VFR asks for 1,000 ft, and an
  overcast at 1,100 ft is legal to take off under and leaves nowhere to fly a circuit. So the
  requirement is `max(1000, TPA + 500)` — 91.155's cloud clearance, applied at the height the
  pattern is actually flown. The pattern altitude comes from NASR via
  `Airport.pattern_altitude_agl_ft` (a height above the field, not an MSL altitude), and only a
  few hundred US fields publish one; the rest take the standard 1,000 ft. Broken is left at the
  regulatory 1,000 ft: there are holes in it and the pilot can see through them.
- **Three statute miles**, from the same report.

The one thing that had to be plumbed for this is `sky_reported`. A clear sky and a source that
does not observe cloud both arrive with no ceiling in them, and they are opposite answers: the
first is VFR, the second is unknown. The model tier reports no cloud at all, so without the flag
a model-only forecast would read as a clear day. It travels with the ceiling and its cover as one
group through `resolve_surface` — a METAR height wearing a TAF's cover code would describe a sky
neither source saw — and on through `Waypoint` to the check. A field with no report at all gets
no weather verdict, so a plan built before pressing **Get weather** reads exactly as it did.

`tools/plot_altimeter_trend.py` is the evidence behind all of this — 24 h of settings for six
Bay Area fields in one request, plotted in inches and in the pressure altitude error they cause.
Over a representative day each field moved about 0.10 inHg (~95 ft), and two fields 60 nm apart
differed by 0.12 inHg (~112 ft) at the same moment. The spread across the area is the better
argument than the drift over time: it is why one field's ATIS should not be used for the next
airport down the route.

### The aloft tier — `engine/aloft.py` + `wx_surface.fetch_aloft`

The surface tier answers what the air is doing at the field. This answers what it is doing at
cruise, which is what decides ground speed, fuel and every time on the log. Same split as
everywhere else: `engine/aloft.py` is pure parsing and interpolation, `fetch_aloft` is the
network.

Not the FD product, deliberately. An FD level is one number for a quarter of a state, and the
wind on the coast is not the wind over the valley twenty miles inland — §4b's reason for keeping
route-wide winds out of the plan. A gridded model is interpolated to the position asked about,
which answers that objection rather than arguing with it. `GET /api/wx/aloft` therefore takes a
**lat/lon, not an ident**: reading the cruise wind off the departure airport would reproduce the
exact mistake.

**The two products are keyed by different altitudes, and it matters.**

- **Temperature → pressure altitude**, exactly, from the level's own pressure. A constant-pressure
  surface *has* a pressure altitude by definition: 850 hPa is 4781 ft PA over the desert and over
  the sea and in a hurricane, and only its geometric height moves. So the temperature profile is
  built with no altimeter setting involved and cannot be wrong by one — and it lands in the
  coordinate the POH charts are read at, and the coordinate `TemperatureSample` already uses.
- **Wind → geopotential height**, the level's true altitude MSL, because that is what a pilot
  holding an indicated altitude on a correct setting is near, and it is what `WindsAloft.at()`
  is called with. The `5000 ft ≈ 850 hPa` table is the ISA mapping; isobaric surfaces move with
  the pressure field, so the height is read from the model rather than assumed.

**Levels below ground are dropped.** A model reports 1000 hPa everywhere, including where the
terrain is at 6000 ft, by extrapolating a fictional atmosphere underneath the mountain. Over
Truckee that discards six levels — 1000 through 850 hPa — and says so in `notes` rather than
interpolating invented air into the bottom of a climb. The ground it compares against is the
`elevation` Open-Meteo returns, which is a 90 m DEM height for the point (5899 ft at Truckee
against a 5900 ft field) and identical across models — not the weather model's own smoothed
terrain, which at 25 km flattens the Sierra by thousands of feet and would keep levels that are
underground in fact.

**Why `models=gfs_seamless` and not `gfs_hrrr`.** Over CONUS, seamless *is* HRRR at 3 km for
roughly the first 48 hours — byte-identical output, verified at 850 hPa over KSQL — and hands off
to GFS beyond it. Pinning HRRR would gain nothing inside that window and would turn a plan made
three days ahead into a refusal rather than coarser data: outside its horizon `gfs_hrrr` returns
nulls. The resolution is worth having where it applies; at Truckee, 700 hPa reads 3.5 °C from
HRRR against 1.8 °C from `gfs_global`, which is real density altitude over exactly the terrain
that punishes getting it wrong. This app is CONUS-only by decision, so HRRR's geographic domain
is not a constraint — its forecast horizon is.

**Feed a plan `temperature_samples()`, not `temperatures()`.** `build_navlog` builds one
temperature curve for the whole route from every sample it can find and would discard a finished
profile, so the samples go in through `Conditions.temperatures_aloft` and merge with the fields'
own METAR temperatures. No conversion is needed on the way in — that is the payoff of keying by
pressure altitude, where a field temperature has to be converted through its own station's
altimeter setting first.

### How it reaches the navlog: two lists, settled against each other

`engine/planwx.py` produces **two lists**, one entry per leg the pilot drew: the navlog rows, and
the weather each of them was planned in. They agree, which is the whole difficulty — each is an
input to the other.

**Both ends of every leg, and the worse of the two.** A leg is not a point. The forecast over the
field it starts at and the forecast over the field it ends at are two different columns of air and
the aeroplane flies through both. Planning on either alone guesses which half of the leg matters;
planning on their average is a wind that was forecast nowhere. So each leg is costed under both
and planned in whichever **costs more fuel** — a plan that comes in early is a good day, a plan
that comes in late is a diversion.

Fuel rather than time, because fuel is what a reserve is measured in and the two can disagree: a
leg flown higher is slower over the ground and cheaper per hour. Only the ground speed is
recomputed to compare the two candidates — the power setting is the same either way, so the rows'
own distance, altitude, TAS and fuel flow are read as they stand.

The column is taken **whole**, its wind and its temperature together, rather than the worst wind
from one end and the worst temperature from the other. Half of one forecast against half of
another describes air neither of them reported — the same objection `resolve_surface` makes about
mixing a METAR's ceiling with a TAF's cover.

**Why it has to loop.** The weather a leg is flown in depends on when the leg is reached; when the
leg is reached depends on the wind it is flown in. Planning once on the departure hour is the
error the whole tier exists to remove — a three-hour leg planned on the 1300Z forecast is not the
leg you fly at 1600Z. So `solve` settles the two lists:

1. Plan with no forecast, to learn roughly when each waypoint is reached.
2. Choose each leg's weather from the forecasts at its two ends, at those times, taking the dearer.
3. Plan again on that weather — which moves the times, and the tops of climb.
4. Re-choose. If nothing changed, the two lists agree and it is done.

**"No more changes" means the choice, not the numbers.** The numbers move by seconds forever. What
has to stop moving is which end of each leg won and which forecast hour it was read at — both
discrete, so settling is a real event rather than a tolerance. Two or three passes is the usual
count; `MAX_PASSES` caps it, and a run that hits the cap still returns its last plan, labelled,
because that is more use than an error.

**One request per waypoint, a window of hours each.** Over the waypoints rather than the leg
midpoints, since adjacent legs share the point between them — N points cover N-1 legs. A *window*
because the loop re-reads the forecast at a different hour on every pass, and going back to the
network each time would put a fetch inside a loop. It costs nothing extra: the Open-Meteo URL is
keyed by day and already carries 48 hours, so `fetch_aloft_series` parses more of one cached
payload. The UI asks for the flight's length plus an hour at each end.

**The engine still takes wind by position.** `solve` hands each chosen column to `build_navlog` at
its leg's midpoint, and `navlog._RouteColumns` files it back under that leg. The obvious key is
the row and it does not work: a forecast wind moves the tops of climb — into a headwind the same
climb covers less ground and tops out sooner — which renumbers the very rows the wind was keyed
to. A point on the earth does not move.

**Containment, not nearest.** `_drawn_leg_of` puts a point on the leg it lies *on*, from the
along-track and cross-track distances `geo.Segment` already computes. Nearest-column is the
tempting shortcut and is wrong at exactly one place: the first ten miles of a 130 nm leg are sixty
miles nearer the *previous* leg's column. The row is still unambiguously on the second leg.

Temperature goes in route-wide as `temperatures_aloft`, merging with the fields' own METARs into
the single curve `build_navlog` builds — a pressure altitude is the coordinate every station
shares, and `TemperatureProfile.from_observations` averages samples that land on the same level.

Where an end has no forecast the other is used; where neither does, the leg falls back to the wind
typed on its row and to calm under that, which is what a plan with nothing entered has always
meant. A wind the pilot typed still wins on its own row, and a half-typed wind takes its other
half from **that leg's** column rather than from a route-wide profile. An end whose wind no
heading can hold the course in is the one thing never chosen despite being the dearest: planning
on it makes the route unbuildable and leaves the pilot looking at an error instead of a plan, so
the other end is used and the leg says so.

This settles §4b's open question about whether a wind should move the top of climb. It does: the
reason it did not was that an FD level is too coarse to trust that far, and a point-resolved model
column is not. The planner gets the forecast on its first pass, since a column belongs to a leg by
where it was forecast and needs no draft lay-out to find its home.

The UI shows the second list under the navlog — both ends' costs, not just the winner's, so a
pilot can see whether the choice was close or obvious.

The navlog says so under the table — a model forecast and a number read off a chart look identical
in a wind column, and they are not the same thing to be flying on.

### 4e. NOTAMs — `engine/notam.py` + `server/notams.py`

A NOTAM search returns everything for a region, and almost none of it is about the flight: a crane
forty miles off track, a closed taxiway at an airport being overflown at 6,500 ft, an airway
closure between FL240 and FL350, a runway shut next Tuesday. Printing all of it is how a briefing
becomes something a pilot skims, and skimming is how the closed runway at the destination gets
missed. So the filter is the feature — the fetch is the easy half.

**Four tests, and all four against the same leg.** A NOTAM has to reach the route corridor
(20 nm either side of track by default), overlap the band of altitude the aeroplane is in *over
that stretch*, and be in force while it is *there*. Testing the three separately against the
whole route keeps a NOTAM that is beside the first leg, at the altitude of the last, and during
the time of neither — so `_against_route` walks leg by leg and a leg has to satisfy all three.
The fourth test is what kind of news it is, which sorts rather than rejects.

The window comes off the navlog, one entry per row: a climb out of a field covers the surface up
to cruise, and the cruise after it covers only cruise. Merged into one entry per drawn leg they
would together claim every altitude over the whole route, and the vertical test would stop
rejecting anything.

**Nothing is rejected on a guess.** A NOTAM with no position, no altitude band or no times cannot
be ruled out on that ground and is kept, with the gap named in its `reasons`. Unstated limits read
open — no lower limit is the surface, no upper limit is unlimited, no end is until further notice
— because every one of those readings keeps a NOTAM rather than drops it. `FL999` is decoded as
unlimited, not as 99,900 ft.

**Distance is to the leg, not to the line through it,** and from the NOTAM's own circle rather
than its centre. A five-mile radius eighteen miles off track reaches a 20 nm corridor; a point at
the same place is clear of it. Measuring by cross-track alone would put a NOTAM two hundred miles
beyond the destination "on the track", since the great circle through a leg does not stop where
the leg does.

**Priority is read from the text.** Anything unrecognised is called operational rather than
information: burying a NOTAM nobody classified is the failure that matters.

#### The source: SkyLink

NOTAMs come from SkyLink's direct API at `data.skylinkapi.com`, authenticated with one licence
key sent as `x-api-key` and read from `RAPID_API_KEY`. (SkyLink also sells the same service
through the RapidAPI marketplace, which uses a different host and an `X-RapidAPI-Key` header;
this is not that channel.) Unconfigured,
`/api/notams` says exactly that instead of returning nothing: **"no NOTAMs" and "no NOTAM service"
look identical on a briefing page and mean opposite things.** For the same reason a partly failed
search is labelled rather than shown as a clean result, and the panel always says how many were
found against how many were kept — "4 of 137" says the filter is working in a way "4 NOTAMs" does
not.

**It is asked by identifier, so both kinds of identifier are needed.** A closed runway is filed
against the aerodrome; a TFR, an MOA or an airspace closure is filed against the ARTCC. A search
that asks only about the airports on the route gets the taxiway closures and misses the restricted
airspace, which is the wrong half to miss. `airports.designators_along_route` returns both: NASR
records the responsible centre for every one of the 12,589 fields in the database, so the centres
come from the aerodromes near the route rather than from airspace boundaries this project does not
yet carry — an approximation that breaks only where a leg clips a corner of a centre with no
airport inside the corridor.

**The aerodromes are found by a chain of circles, and the spacing is what makes it a corridor.**
Circles of radius R every R nm cover everything within `R·√3/2` of the track, the thin spot being
where two adjacent circles cross, so the search radius is set from the corridor width
(`corridor / (√3/2)` ≈ 23 nm for a 20 nm corridor) rather than equal to it. Searching at 20 nm
left scalloped gaps, and an aerodrome 19 nm off track halfway between two samples was never asked
about. `tests/test_notam_fetch.py` samples the whole corridor edge against the chain; it caught two
holes in the thinning that drops circles sitting on top of one another, and a point is now dropped
only when what follows it still lands within one spacing of the circle before it.

**The budget decides the order.** The free tier is 1,000 requests a month and each identifier is
one request, so `MAX_DESIGNATORS` caps a route at 40 — 25 briefings a month at the cap. A 150 nm
route in busy airspace has 60 to 90 identifiers within reach, so the cap is the normal case, not the
edge case, and what it cuts matters. Identifiers are asked in this order:

1. the route's own aerodromes — departure, stops, destination;
2. the ARTCCs;
3. every other aerodrome, nearest the track first.

The first version walked from departure to destination and cut at the cap, which on every busy
route dropped the destination — the one aerodrome a pilot is certain to land at. The cut is always
reported as an incomplete search, never applied silently.

**The raw text supplies what the fields leave out.** SkyLink documents its fields as "NOTAM ID,
type, location, effective time, expiration time, and body" — nothing about position or altitude.
The raw ICAO text has both, in the Q-line, so `parse_q_line` decodes it underneath whatever came
structurally: centre, radius, lower and upper limits, and the B/C start and end. That is what makes
four-dimensional filtering possible from an identifier-only source.

**An unreadable reply is an outage, never an empty airport.** `parse_skylink` raises
`UnrecognisedPayload` on any reply it cannot read, including a list whose every record is
unreadable; an empty list comes back only for a reply that was recognisably an empty NOTAM list,
since a small field often has none. The fetcher records an unreadable reply as a failure for that
identifier, and raises if every identifier failed — counting query failures on their own, so that
the budget notice in the same failure list can never turn a total outage into an empty briefing.

**The decoder is written to SkyLink's published description and has not yet met a live
response** — the service returns 401 without a key. The first real reply should be checked against
`tests/test_notam_skylink.py`; if the field names differ, the reply fails loudly as unreadable
rather than showing an empty panel.

---

## 4e. Raster charts — `engine/tiff_reader.py`, `engine/charts.py`, `engine/chart_render.py` — [built]

The sectional and terminal area charts, drawn under the route. The FAA publishes each
as one palette GeoTIFF (30–80 MB, LZW, Lambert Conformal Conic) with a `.tfw` world file
and an `.htm` of FGDC metadata. Unzip a download into `data/charts/sectional/` or
`data/charts/tac/` and it appears in the map's **Layers** menu; nothing is committed
(see `data/charts/README.txt`).

Three modules, one job each:

- **`tiff_reader`** parses the TIFF directory and GeoTIFF keys directly — pixel scale,
  tie point, projection parameters — in a few `struct` calls, and implements the Lambert
  projection both ways (Snyder 15-1 to 15-11). GDAL would do the same and cost a 100 MB
  dependency that Pyodide does not have. Pixels are decoded by Pillow, which opens the
  200-megapixel sectional in about a second.
- **`charts`** walks the folder, and reads each chart's footprint from the tags and its
  edition dates from the `.htm` (`Beginning_Date` / `Ending_Date`; VFR charts run a 56-day
  cycle). Each chart becomes a `currency.Dataset`, so the sidebar's currency list shows it
  going stale beside NASR and the magnetic model. The `.htm` and `.tif` are checked against
  each other by image size, the way NASR's README is checked against the airport database.
- **`chart_render`** resamples the Lambert image into Web Mercator XYZ tiles, which is the
  only raster MapLibre draws. Every tile pixel is projected into chart pixel coordinates
  and the nearest one taken — vectorised, a few milliseconds a tile. Zoomed out, tiles
  sample a box-filtered pyramid built lazily so thin lines do not alias away. Outside the
  image is transparent, so the basemap shows past the chart's edge. `TileStore` keeps
  tiles as PNG under `data/charts/tiles/`; the dev server fills it on demand
  (`/api/charts/tiles/…`), `make charts` fills all of it for the offline bundle, and
  neither renders a tile twice.

The choice that matters: **tiles are rendered outside the browser.** Decoding a
200-megapixel TIFF and reprojecting it in JavaScript would hold several hundred megabytes
in a tab and take tens of seconds. Pre-rendered PNG tiles load lazily and are what the PWA
bundle will ship. Pillow joins the runtime dependencies for this — it is in Pyodide, so the
rule from §1 holds, and on a deployed server it is never imported at all.

### The deployed case — what is committed and why

`data/charts/` holds the FAA downloads, which are **not** committed: 30–80 MB each, reissued
every 56 days, and useless to a server that cannot afford to decode one. `data/charts/tiles/`
holds the pyramid built from them, which **is** committed, on the same reasoning as
`data/aero/airports.sqlite` — it is the build output the app ships with, and the only form in
which a chart reaches a browser anywhere but this desktop.

That makes the two run modes genuinely different, and `charts.available()` is where the fork
lives:

| | Desktop | Deployed |
|---|---|---|
| Charts listed from | GeoTIFF headers | `tiles/manifest.json` |
| Tiles come from | `/api/charts/tiles/…`, rendered on demand | `/data/charts/tiles/…`, static files |
| Peak memory | 410 MB for a sectional | nothing decoded, ever |

The manifest carries what the headers would have said — footprint, zoom range, edition dates —
so the layer menu and the currency list read identically either way. Its zoom range is read
back off the tile directories rather than taken from the chart, because `make charts MAXZOOM=`
caps the pyramid and a source advertising a level with no tiles behind it shows as holes.

**Tiles are palette PNGs**, which is a third the size of RGBA and matters when every byte is
committed and cloned on each deploy. A tile cut at the chart's own scale holds only palette
colours, so its palette is exact and nothing is lost; a tile off a pyramid level holds blended
colours and is quantised to 255, which is the one lossy step in the pipeline and sits on an
image that is already a reduction. The palette is written no longer than the tile needs, since
most tiles hold a handful of colours and a padded 256-entry table would be 768 bytes on each.

Zoom is where the size is, and it is the dial to turn when the committed tree gets too big.
Each level has four times the tiles of the one below, so the top one or two are most of the
bytes — for the Bay Area set, the full pyramid is 93 MB and two levels off the top is 33 MB.
MapLibre overzooms past whatever the manifest advertises, so a capped chart goes soft rather
than blank, and `make charts MAXZOOM=` is what sets the cap.

What is committed now is the sectional to z10 and the terminal area charts to z12. The
sectional is crisp at the zoom a route is actually planned at and blocky if you push past it;
the terminal charts are within a level of native over the ground where you would.

| Committed | Tiles | Size |
|---|---|---|
| Sectional z5–10, TAC and Flyway z5–12 | 2,381 | 33 MB |
| Sectional z5–11, TAC and Flyway z5–13 | 8,968 | 93 MB |

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
