/* my_e6b — route editor and live navigation log.
 *
 * This file talks to the engine through exactly three operations: search for
 * an airport, list airports in view, and plan a route. In the packaged PWA
 * the same calls go to Pyodide instead of HTTP, so `api` below is the only
 * thing that changes between the two run modes. Keep planning logic out of
 * here — it belongs in engine/, or the desktop and iPad builds will drift.
 */

'use strict';

// --- the entire server boundary ----------------------------------------

const api = {
  async searchAirports(query) {
    const r = await fetch(`/api/airports/search?q=${encodeURIComponent(query)}`);
    return r.ok ? r.json() : [];
  },
  async airportsInView(bounds, minRunway) {
    const p = new URLSearchParams({
      south: bounds.getSouth(), west: bounds.getWest(),
      north: bounds.getNorth(), east: bounds.getEast(),
    });
    if (minRunway) p.set('min_runway_ft', minRunway);
    const r = await fetch(`/api/airports/bbox?${p}`);
    return r.ok ? r.json() : [];
  },
  async vfrWaypointsInView(bounds) {
    const p = new URLSearchParams({
      south: bounds.getSouth(), west: bounds.getWest(),
      north: bounds.getNorth(), east: bounds.getEast(),
    });
    const r = await fetch(`/api/vfr-waypoints/bbox?${p}`);
    return r.ok ? r.json() : [];
  },
  async status() {
    const r = await fetch('/api/status');
    return r.json();
  },
  /** The raster charts on disk: where each sits, its tile URL, and its dates. */
  async charts() {
    const r = await fetch('/api/charts');
    return r.ok ? r.json() : { charts: [] };
  },
  /** Surface weather at one field, resolved from METAR, TAF and model.
   *
   *  No `time`: there is no departure time in the form, so this is the
   *  weather now. An engine-level refusal ("no observation for this field")
   *  comes back as ok:false rather than as an HTTP error, so both shapes are
   *  normalised here into something with an `error` on it or not. */
  async surface(ident, isoTime) {
    const at = isoTime ? `&time=${encodeURIComponent(isoTime)}` : '';
    const r = await fetch(`/api/wx/surface?ident=${encodeURIComponent(ident)}${at}`);
    const body = await r.json().catch(() => null);
    if (!r.ok) return { error: (body && body.detail) || `no weather for ${ident}` };
    if (body && body.ok === false) return { error: body.error };
    return body;
  },

  async aloftSeries(lat, lon, isoTime, hours) {
    const p = new URLSearchParams({ lat, lon, hours });
    if (isoTime) p.set('time', isoTime);
    const r = await fetch(`/api/wx/aloft/series?${p}`);
    const body = await r.json().catch(() => null);
    if (!r.ok) return { error: (body && body.detail) || 'no forecast for this point' };
    if (body && body.ok === false) return { error: body.error };
    return body;
  },

  /** The forecast window over every point of the route, in one request.
   *
   *  One call rather than one per waypoint. The model's free tier counts
   *  requests per IP address, and a deployed instance shares its address with
   *  strangers -- a route's worth of separate calls is what got this refused
   *  with a 429 in the first place, which showed up as a plan quietly falling
   *  back to the standard atmosphere.
   *
   *  Answers positionally: `points[n]` is the forecast over `points[n]` of
   *  the argument, and a point the model could not answer for keeps its slot
   *  with an error on it. A whole-batch failure is normalised to that same
   *  shape, so the caller has one case to handle rather than two. */
  async aloftSeriesMany(points, isoTime, hours) {
    const p = new URLSearchParams({
      lat: points.map((w) => w.lat).join(','),
      lon: points.map((w) => w.lon).join(','),
      hours,
    });
    if (isoTime) p.set('time', isoTime);
    const r = await fetch(`/api/wx/aloft/series/batch?${p}`);
    const body = await r.json().catch(() => null);
    const failed = (error) => points.map(() => ({ error }));
    if (!r.ok) return failed((body && body.detail) || 'no forecast for this route');
    if (!body || body.ok === false) {
      return failed((body && body.error) || 'no forecast for this route');
    }
    // A short list would silently shift every forecast onto the wrong
    // waypoint, so the length is the contract and a mismatch is a failure.
    if (!Array.isArray(body.points) || body.points.length !== points.length) {
      return failed('the forecast did not line up with the route');
    }
    return body.points.map((point) => (point && point.ok
      ? point
      : { error: (point && point.error) || 'no forecast for this point' }));
  },

  async notams(body) {
    const r = await fetch('/api/notams', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const answer = await r.json().catch(() => null);
    return answer || { ok: false, error: 'the NOTAM search did not answer' };
  },

  async plan(body) {
    const r = await fetch('/api/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.json();
  },
  /** The navlog as a paper-nav-log CSV. Returns the file, not JSON: the
   *  server owns the format and the filename, so there is nothing to build
   *  here beyond saving what comes back. */
  async navlogCsv(body) {
    const r = await fetch('/api/navlog.csv', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) return { error: (await r.json()).error || 'could not export' };
    const disposition = r.headers.get('Content-Disposition') || '';
    const named = /filename="([^"]+)"/.exec(disposition);
    return { blob: await r.blob(), filename: named ? named[1] : 'navlog.csv' };
  },
  /** The route as KML, the same way: a file, named by the server. */
  async routeKml(body) {
    const r = await fetch('/api/route.kml', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) return { error: (await r.json()).error || 'could not export' };
    const disposition = r.headers.get('Content-Disposition') || '';
    const named = /filename="([^"]+)"/.exec(disposition);
    return { blob: await r.blob(), filename: named ? named[1] : 'route.kml' };
  },
  /** The waypoints in a KML/KMZ file. The bytes go up base64 in JSON, like
   *  every other request here; the server parses and resolves airports. */
  async importRoute(file) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    const r = await fetch('/api/route/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: file.name, content_base64: btoa(binary) }),
    });
    return r.json();
  },
  async consistency(body) {
    const r = await fetch('/api/consistency', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.json();
  },

  async densityAltitude({ elevationFt, oatC, altimeterInhg }) {
    const p = new URLSearchParams({
      elevation_ft: elevationFt, oat_c: oatC, altimeter_inhg: altimeterInhg,
    });
    const r = await fetch(`/api/e6b/density-altitude?${p}`);
    return r.json();
  },
  async variation({ lat, lon }) {
    const p = new URLSearchParams({ lat, lon });
    const r = await fetch(`/api/e6b/variation?${p}`);
    return r.json();
  },
  async weightBalance(stations) {
    const r = await fetch('/api/e6b/weight-balance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stations }),
    });
    return r.json();
  },
};

// Phase colours, single source of truth for the map paint expression. Kept in
// step with --climb / --cruise / --descent in style.css: climb blue, cruise
// green, descent orange.
const PHASE_COLOUR = {
  climb: '#4da3ff',
  cruise: '#7ee081',
  descent: '#ffa657',
};

const SEGMENT_TYPES = ['climb', 'cruise', 'descent'];

// --- state --------------------------------------------------------------

/** The route. Order is the flight order; index 0 departs, last arrives. */
let route = [];
let markers = [];
let lastPlan = null;
/** "manual" -- the default -- takes the profile from each waypoint's segment
 *  type; "hybrid" lets the planner place TOC/TOD instead, around whatever
 *  events the pilot pinned to each leg. Kept in step with the toggle's pressed
 *  state and the altitude field's disabled state in the markup, which start on
 *  the same mode. */
let planningMode = 'manual';

/** A stable identity for one of the pilot's waypoints.
 *
 *  What a leg is named by -- "<id>><id>" -- so that its events and the edits
 *  typed on it survive the route being edited around it, and so that a saved
 *  mission names the same legs when it is loaded again. */
function newId() {
  return (crypto.randomUUID?.() || `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`)
    .slice(0, 12);
}

/** Give every pilot waypoint an id. Idempotent; generated points have none. */
function ensureIds() {
  for (const w of route) {
    if (!w.generated && !w.id) w.id = newId();
  }
}

/** The legs the pilot drew: consecutive pilot waypoints, TOC/TOD skipped. */
function pilotLegs() {
  const legs = [];
  let from = null;
  route.forEach((w, index) => {
    if (w.generated) return;
    if (from) legs.push({ from, to: w, index, key: `${from.id}>${w.id}` });
    from = w;
  });
  return legs;
}

/** "SUNOL → VPCLB" for a leg key, or the key itself if the leg is gone. */
function legName(key) {
  const leg = pilotLegs().find((l) => l.key === key);
  return leg ? `${leg.from.name} → ${leg.to.name}` : key;
}

/** Where along a leg a point falls, 0..1, and the point on the leg there.
 *  Planar, like `distanceToSegment`: it only has to snap a click; the engine
 *  projects the saved point onto the great circle itself. */
function projectOntoLeg(lngLat, a, b) {
  const k = Math.cos((a.lat * Math.PI) / 180);
  const ax = a.lon * k, ay = a.lat;
  const dx = b.lon * k - ax, dy = b.lat - ay;
  const lengthSq = dx * dx + dy * dy;
  const t = lengthSq === 0 ? 0
    : Math.max(0, Math.min(1, ((lngLat.lng * k - ax) * dx + (lngLat.lat - ay) * dy) / lengthSq));
  return {
    t,
    lat: +(a.lat + t * (b.lat - a.lat)).toFixed(5),
    lon: +(a.lon + t * (b.lon - a.lon)).toFixed(5),
  };
}
/** Findings from the last consistency check, or null if it has not been run
 *  or the route changed under it. */
let consistencyReport = null;

/** Pressure and density altitude at each waypoint's own elevation, as the
 *  engine computed them on the last plan, indexed by route position. Derived
 *  and never sent back: the inputs live on the waypoints. */
let fieldAir = [];

/** What the last "Get weather" fetched, keyed by waypoint name.
 *
 *  Kept beside the route rather than on it: the values it filled in are the
 *  pilot's to edit afterwards, and once edited the provenance line would be
 *  claiming a METAR said something it did not. So each entry also remembers
 *  what it wrote, and the line disappears from any box that has since moved.
 */
let fieldWx = new Map();

/** A window of forecast hours over each waypoint the pilot drew.
 *
 *  Over the waypoints, not the leg midpoints, because a leg is costed at both
 *  of its ends and flown in whichever end costs more -- see `engine/planwx`.
 *  Two adjacent legs share the waypoint between them, so N waypoints cover
 *  N-1 legs with N requests rather than 2(N-1).
 *
 *  A window rather than an hour because the engine settles the weather and
 *  the log against each other, and each pass reads the forecast at the hour
 *  that pass says the leg is reached. Sending the window means the loop turns
 *  without going back to the network.
 *
 *  Sent with the plan, not stored on the route: a forecast belongs to a place
 *  and a time, and the moment the route changes it is a forecast for
 *  somewhere else.
 */
let forecasts = [];

/** Where the forecasts came from, for the line under the navlog. */
let forecastWx = null;

/** Drop the forecasts. The route they were fetched for is gone. */
function clearForecasts() {
  if (!forecasts.length && forecastWx === null) return;
  forecasts = [];
  forecastWx = null;
}

// Matches the server's cap on one forecast request. A light single's day is
// a few hours; asking for more would be a mistyped number, not a longer trip.
const MAX_FORECAST_HOURS = 12;

const $ = (id) => document.getElementById(id);
const setStatus = (text) => { $('status').textContent = text; };

// --- map ----------------------------------------------------------------

const map = new maplibregl.Map({
  container: 'map',
  // No tile server: the basemap is local GeoJSON, so this works offline.
  style: {
    version: 8,
    sources: {},
    layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#0d1218' } }],
  },
  center: [-122.1, 37.2],
  zoom: 7,
  attributionControl: false,
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
map.addControl(new maplibregl.ScaleControl({ unit: 'nautical' }), 'bottom-right');

map.on('load', async () => {
  await addBasemap();
  await addChartLayers();
  addRouteLayers();
  bindMapInteractions();
  refreshAirportLayer();
  renderDataCurrency();
  setStatus('Ready — offline');
});

/** Show how old the bundled data is: NASR runs out in a month, the
 *  magnetic model in five years, and offline there is nothing to warn us. */
async function renderDataCurrency() {
  const list = $('data-currency');
  let datasets = [];
  try {
    datasets = (await api.status()).data || [];
  } catch {
    return;
  }
  list.innerHTML = '';
  for (const d of datasets) {
    const li = document.createElement('li');
    if (d.expired) li.classList.add('expired');
    const when = d.effective
      ? `${d.effective}${d.expired ? ' — expired' : ` — ${d.days_remaining} d left`}`
      : 'date unknown';
    li.innerHTML = `<span>${d.label}</span><span class="date">${when}</span>`;
    if (d.note) {
      const note = document.createElement('span');
      note.className = 'note';
      note.textContent = d.note;
      li.appendChild(note);
    }
    li.title = d.expires ? `Effective ${d.effective}, expires ${d.expires}` : '';
    list.appendChild(li);
  }
}

async function addBasemap() {
  for (const name of ['lakes', 'coastline', 'states', 'highways']) {
    const data = await (await fetch(`/data/basemap/${name}.geojson`)).json();
    map.addSource(name, { type: 'geojson', data });
  }
  map.addLayer({
    id: 'lakes-fill', type: 'fill', source: 'lakes',
    paint: { 'fill-color': '#0f1c2b' },
  });
  // Under the borders and coastline so those stay readable, and warm brown so
  // roads never read as water. No symbol layer for the route numbers: that
  // needs a `glyphs` URL, which is a network dependency this app cannot have.
  map.addLayer({
    id: 'highways-line', type: 'line', source: 'highways',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': '#3a3226',
      'line-width': ['interpolate', ['linear'], ['zoom'], 5, 0.4, 10, 1.6],
      // Two thousand lines are noise at continental zoom.
      'line-opacity': ['interpolate', ['linear'], ['zoom'], 4, 0, 6, 0.85],
    },
  });
  // A transparent fat line over the drawn one, purely so the pointer can hit a
  // road that is under two pixels wide. Invisible layers are still returned by
  // feature queries, which is the whole trick.
  map.addLayer({
    id: 'highways-hit', type: 'line', source: 'highways',
    paint: { 'line-color': '#000', 'line-opacity': 0, 'line-width': 12 },
  });
  map.addLayer({
    id: 'states-line', type: 'line', source: 'states',
    paint: { 'line-color': '#243547', 'line-width': 1 },
  });
  map.addLayer({
    id: 'coastline-line', type: 'line', source: 'coastline',
    paint: { 'line-color': '#2f4459', 'line-width': 1.2 },
  });
  await addAirspace();
}

// Sectional convention, adapted to a dark background: Class B solid blue,
// Class C solid magenta, Class D dashed blue. Class E is not in the data --
// see tools/build_airspace.py for why.
const AIRSPACE_COLOUR = ['match', ['get', 'class'], 'C', '#b0568f', '#4a7fc1'];

async function addAirspace() {
  const data = await (await fetch('/data/aero/airspace.geojson')).json();
  map.addSource('airspace', { type: 'geojson', data });

  // A wash rather than a fill: Class B shelves stack, and at full opacity the
  // overlaps would read as darker airspace that does not exist.
  map.addLayer({
    id: 'airspace-fill', type: 'fill', source: 'airspace',
    paint: {
      'fill-color': AIRSPACE_COLOUR,
      'fill-opacity': ['interpolate', ['linear'], ['zoom'], 5, 0, 7, 0.07],
    },
  });
  map.addLayer({
    id: 'airspace-line', type: 'line', source: 'airspace',
    paint: {
      'line-color': AIRSPACE_COLOUR,
      'line-width': ['interpolate', ['linear'], ['zoom'], 5, 0.6, 10, 1.6],
      'line-opacity': ['interpolate', ['linear'], ['zoom'], 4, 0, 6, 0.75],
      // Only Class D is dashed. MapLibre cannot vary dashes per feature, so
      // the D outline is a second layer rather than a data expression.
      'line-dasharray': [1, 0],
    },
    filter: ['!=', ['get', 'class'], 'D'],
  });
  map.addLayer({
    id: 'airspace-line-d', type: 'line', source: 'airspace',
    paint: {
      'line-color': AIRSPACE_COLOUR,
      'line-width': ['interpolate', ['linear'], ['zoom'], 5, 0.6, 10, 1.4],
      'line-opacity': ['interpolate', ['linear'], ['zoom'], 4, 0, 6, 0.75],
      'line-dasharray': [3, 2],
    },
    filter: ['==', ['get', 'class'], 'D'],
  });
}

// --- raster charts -------------------------------------------------------
//
// FAA sectionals and terminal area charts, as Web Mercator tiles the server
// renders from the GeoTIFFs under data/charts/ (engine/chart_render.py). The
// menu lists whatever is on disk, grouped by series. A chart is drawn under
// the airspace outlines and the route so the plan stays legible on top of
// it, and above the vector basemap, which it replaces where it covers.
// Which charts are on is remembered per browser: a pilot flying the Bay
// Area wants the TAC every time.

const CHART_LAYERS_KEY = 'charts.visible';
// key ("tac/san_francisco_tac") -> { chart, id } in menu (and draw) order.
const chartLayers = new Map();

function loadVisibleCharts() {
  try {
    return new Set(JSON.parse(localStorage.getItem(CHART_LAYERS_KEY) || '[]'));
  } catch {
    return new Set();
  }
}

function saveVisibleCharts(keys) {
  try { localStorage.setItem(CHART_LAYERS_KEY, JSON.stringify([...keys])); } catch { /* private mode */ }
}

async function addChartLayers() {
  let charts = [];
  try {
    charts = (await api.charts()).charts || [];
  } catch {
    charts = [];
  }
  const visible = loadVisibleCharts();
  for (const chart of charts) {
    const key = `${chart.kind}/${chart.slug}`;
    const id = `chart-${chart.kind}-${chart.slug}`;
    map.addSource(id, {
      type: 'raster',
      tiles: [chart.tiles],
      tileSize: 256,
      minzoom: chart.min_zoom,
      // Past this the source's own pixels are simply magnified; MapLibre
      // overzooms the last level rather than asking for more.
      maxzoom: chart.max_zoom,
      // So no tile is requested for the empty ocean around the chart.
      bounds: chart.bounds,
    });
    map.addLayer({
      id, type: 'raster', source: id,
      layout: { visibility: visible.has(key) ? 'visible' : 'none' },
      paint: { 'raster-fade-duration': 0 },
    }, 'airspace-fill');
    chartLayers.set(key, { chart, id });
  }
  renderLayerMenu();
}

function setChartVisible(key, on) {
  const entry = chartLayers.get(key);
  if (!entry) return;
  map.setLayoutProperty(entry.id, 'visibility', on ? 'visible' : 'none');
  const visible = loadVisibleCharts();
  if (on) visible.add(key); else visible.delete(key);
  saveVisibleCharts(visible);
}

/** The chart's dates, the way the sidebar's currency list words them. */
function chartDateText(chart) {
  if (!chart.effective && !chart.expires) return 'edition dates unknown';
  const span = `${chart.effective || '?'} → ${chart.expires || '?'}`;
  if (chart.expired) return `${span} — expired`;
  if (chart.days_remaining != null) return `${span} — ${chart.days_remaining} d left`;
  return span;
}

function renderLayerMenu() {
  const list = $('layers-list');
  list.innerHTML = '';
  $('layers-empty').hidden = chartLayers.size > 0;
  const visible = loadVisibleCharts();
  let lastKind = null;
  for (const [key, { chart }] of chartLayers) {
    if (chart.kind !== lastKind) {
      const heading = document.createElement('div');
      heading.className = 'kind';
      heading.textContent = chart.kind_label;
      list.appendChild(heading);
      lastKind = chart.kind;
    }
    const label = document.createElement('label');
    if (chart.expired) label.classList.add('expired');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = visible.has(key);
    input.addEventListener('change', () => setChartVisible(key, input.checked));
    const text = document.createElement('span');
    text.textContent = chart.name;
    const date = document.createElement('span');
    date.className = 'date';
    date.textContent = chartDateText(chart);
    text.appendChild(date);
    if (chart.note) {
      const note = document.createElement('span');
      note.className = 'note';
      note.textContent = chart.note;
      text.appendChild(note);
    }
    label.title = chart.expires
      ? `Effective ${chart.effective}, expires ${chart.expires}`
      : chart.name;
    label.append(input, text);
    list.appendChild(label);
  }
}

$('layers-toggle').addEventListener('click', () => {
  const panel = $('layers');
  panel.hidden = !panel.hidden;
  $('layers-toggle').setAttribute('aria-expanded', String(!panel.hidden));
});

function addRouteLayers() {
  map.addSource('airports', { type: 'geojson', data: emptyCollection() });
  map.addLayer({
    id: 'airports-dot', type: 'circle', source: 'airports',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 6, 2, 10, 4.5],
      'circle-color': '#5b7e9e', 'circle-opacity': 0.9,
    },
  });
  // No MapLibre symbol layer for airport identifiers. `text-field` requires a
  // `glyphs` URL serving font PBFs, which is a network dependency this app
  // cannot have. Labels are drawn as HTML instead -- see renderAirportLabels.

  // Published VFR waypoints, drawn magenta to match the flag symbol used on
  // sectionals. They sit above the airport dots because they are the points a
  // VFR route should actually be built from.
  map.addSource('vfr', { type: 'geojson', data: emptyCollection() });
  map.addLayer({
    id: 'vfr-dot', type: 'circle', source: 'vfr',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 6, 2.5, 10, 5],
      'circle-color': '#d24ba0',
      'circle-stroke-color': '#2a1020',
      'circle-stroke-width': 1,
    },
  });

  map.addSource('route', { type: 'geojson', data: emptyCollection() });
  map.addLayer({
    id: 'route-line', type: 'line', source: 'route',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-width': 3,
      'line-color': [
        'match', ['get', 'phase'],
        'climb', PHASE_COLOUR.climb,
        'descent', PHASE_COLOUR.descent,
        PHASE_COLOUR.cruise,
      ],
    },
  });
}

const emptyCollection = () => ({ type: 'FeatureCollection', features: [] });

function bindMapInteractions() {
  map.on('moveend', refreshAirportLayer);

  // Clicking an airport dot offers to add it to the route.
  map.on('click', 'airports-dot', (event) => {
    if (eventPick) return;
    const f = event.features[0];
    showAirportPopup(f.geometry.coordinates, f.properties);
  });
  map.on('click', 'vfr-dot', (event) => {
    if (eventPick) return;
    const f = event.features[0];
    showVfrWaypointPopup(f.geometry.coordinates, f.properties);
  });

  for (const layer of ['airports-dot', 'vfr-dot']) {
    map.on('mouseenter', layer, () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', layer, () => { map.getCanvas().style.cursor = ''; });
  }

  // Naming the road under the cursor. Hover rather than click, because a click
  // on the map inserts a waypoint and a road is not somewhere you fly to.
  map.on('mousemove', 'highways-hit', showHighwayTooltip);
  map.on('mouseleave', 'highways-hit', hideHighwayTooltip);

  // Same idea for airspace, where the floor and ceiling are the whole point:
  // knowing a Class B shelf is overhead means nothing without knowing at what
  // altitude it starts.
  map.on('mousemove', 'airspace-fill', showAirspaceTooltip);
  map.on('mouseleave', 'airspace-fill', hideAirspaceTooltip);

  // Clicking empty map inserts a free waypoint into the nearest leg.
  map.on('click', (event) => {
    // Placing a leg event takes the click wherever it lands, dots included.
    if (eventPick) { placeEvent(event.lngLat); return; }
    const hits = map.queryRenderedFeatures(event.point, {
      layers: ['airports-dot', 'vfr-dot'],
    });
    if (hits.length) return;
    if (route.length < 2) return;
    insertWaypointAt(event.lngLat);
  });
}

async function refreshAirportLayer() {
  if (!map.getSource('airports')) return;
  const zoom = map.getZoom();
  // Thin the display when zoomed out; fifteen thousand dots is not a map.
  const minRunway = zoom < 7 ? 5000 : zoom < 9 ? 3000 : null;
  const airports = await api.airportsInView(map.getBounds(), minRunway);
  map.getSource('airports').setData({
    type: 'FeatureCollection',
    features: airports.map((a) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [a.lon, a.lat] },
      properties: a,
    })),
  });
  renderAirportLabels(airports, zoom);
  refreshVfrLayer(zoom);
}

async function refreshVfrLayer(zoom) {
  if (!map.getSource('vfr')) return;
  // There are only 663 of these nationally, so no thinning is needed the way
  // twelve thousand airports need it. They are hidden right out at low zoom
  // only because a scatter of unlabelled dots is noise, not information.
  if (zoom < VFR_MIN_ZOOM) {
    map.getSource('vfr').setData(emptyCollection());
    renderVfrLabels([], zoom);
    return;
  }
  const waypoints = await api.vfrWaypointsInView(map.getBounds());
  map.getSource('vfr').setData({
    type: 'FeatureCollection',
    features: waypoints.map((w) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [w.lon, w.lat] },
      properties: w,
    })),
  });
  renderVfrLabels(waypoints, zoom);
}

/** Identifiers, as HTML rather than a MapLibre symbol layer.
 *  Only close in, and capped, because these are real DOM nodes. */
let labelMarkers = [];
let vfrLabelMarkers = [];
const LABEL_MIN_ZOOM = 8;
const LABEL_MAX_COUNT = 60;
const VFR_MIN_ZOOM = 7;

function renderLabels(points, className) {
  return points.map((point) => {
    const element = document.createElement('div');
    element.className = className;
    element.textContent = point.ident;
    return new maplibregl.Marker({ element, anchor: 'top' })
      .setLngLat([point.lon, point.lat])
      .addTo(map);
  });
}

function renderAirportLabels(airports, zoom) {
  labelMarkers.forEach((m) => m.remove());
  labelMarkers = [];
  if (zoom < LABEL_MIN_ZOOM) return;
  labelMarkers = renderLabels(airports.slice(0, LABEL_MAX_COUNT), 'airport-label');
}

function renderVfrLabels(waypoints, zoom) {
  vfrLabelMarkers.forEach((m) => m.remove());
  vfrLabelMarkers = [];
  if (zoom < LABEL_MIN_ZOOM) return;
  vfrLabelMarkers = renderLabels(waypoints.slice(0, LABEL_MAX_COUNT), 'vfr-label');
}

/** "I-5", "US-101", "State route 1". Natural Earth stores the bare route
 *  number, so the prefix has to come from `level`. State routes get no state
 *  code because the source does not carry one. */
function highwayName(props) {
  const number = props.name;
  if (!number) return null;
  if (props.level === 'Interstate') return `I-${number}`;
  if (props.level === 'Federal') return `US-${number}`;
  if (props.level === 'State') return `State route ${number}`;
  return `Route ${number}`;
}

let highwayTooltip = null;

function showHighwayTooltip(event) {
  // An airport or waypoint sitting on top wins: those are clickable and the
  // tooltip would cover what the user is reaching for.
  const dots = map.queryRenderedFeatures(event.point, {
    layers: ['airports-dot', 'vfr-dot'],
  });
  if (dots.length) return hideHighwayTooltip();

  const name = highwayName(event.features[0].properties);
  if (!name) return hideHighwayTooltip();

  if (!highwayTooltip) {
    highwayTooltip = new maplibregl.Popup({
      closeButton: false, closeOnClick: false, className: 'highway-tip',
      offset: 10, anchor: 'bottom',
    });
  }
  highwayTooltip.setLngLat(event.lngLat).setText(name).addTo(map);
}

function hideHighwayTooltip() {
  if (highwayTooltip) highwayTooltip.remove();
}

/** "SFC–2500" or "1000–7000", in feet. The ceiling is always MSL in the
 *  source; the floor is either MSL or the surface, and that difference is the
 *  one a pilot actually acts on. */
function airspaceBand(props) {
  const floor = props.lower_code === 'SFC' ? 'SFC' : props.lower;
  return `${floor}–${props.upper} ft`;
}

let airspaceTooltip = null;

function showAirspaceTooltip(event) {
  const dots = map.queryRenderedFeatures(event.point, {
    layers: ['airports-dot', 'vfr-dot'],
  });
  if (dots.length) return hideAirspaceTooltip();

  // Shelves overlap, so report every layer under the cursor rather than the
  // topmost one. Stacked floors are exactly what a VFR pilot is checking.
  const seen = new Set();
  const rows = [];
  for (const f of map.queryRenderedFeatures(event.point, { layers: ['airspace-fill'] })) {
    const line = `Class ${f.properties.class} · ${airspaceBand(f.properties)}`;
    if (seen.has(line)) continue;
    seen.add(line);
    rows.push({ line, name: f.properties.name });
  }
  if (!rows.length) return hideAirspaceTooltip();

  const node = document.createElement('div');
  node.innerHTML = rows
    .map((r) => `<div class="airspace-row"><b>${r.line}</b><br>${r.name}</div>`)
    .join('');

  if (!airspaceTooltip) {
    airspaceTooltip = new maplibregl.Popup({
      closeButton: false, closeOnClick: false, className: 'airspace-tip',
      offset: 12, anchor: 'bottom',
    });
  }
  airspaceTooltip.setLngLat(event.lngLat).setDOMContent(node).addTo(map);
}

function hideAirspaceTooltip() {
  if (airspaceTooltip) airspaceTooltip.remove();
}

function showVfrWaypointPopup(coordinates, props) {
  const node = document.createElement('div');
  node.innerHTML =
    `<div class="popup-title vfr">${props.ident}</div>` +
    `<div class="popup-meta">Published VFR checkpoint` +
    `${props.region ? ` · ${props.region}` : ''}<br>` +
    `What it marks is printed on the sectional</div>` +
    `<div class="popup-actions"><button type="button">Add to route</button></div>`;
  const popup = new maplibregl.Popup({ closeButton: false })
    .setLngLat(coordinates).setDOMContent(node).addTo(map);
  node.querySelector('button').addEventListener('click', () => {
    addWaypoint({
      name: props.ident,
      lat: props.lat, lon: props.lon,
      kind: 'vfr_waypoint',
      // Null, not zero: a VFR waypoint has no elevation, and the navlog
      // refuses a route whose endpoints lack one. That refusal is correct --
      // you cannot depart from a bridge -- and it depends on this staying null.
      elevation_ft: null,
      label: props.label,
    });
    popup.remove();
  });
}

function showAirportPopup(coordinates, props) {
  const runway = props.longest_runway_ft
    ? `${Math.round(props.longest_runway_ft)} ft runway` : 'runway length unknown';
  const node = document.createElement('div');
  node.innerHTML =
    `<div class="popup-title">${props.ident}</div>` +
    `<div class="popup-meta">${props.name}<br>` +
    `${Math.round(props.elevation_ft)} ft elev · ${runway}</div>` +
    `<div class="popup-actions"><button type="button">Add to route</button></div>`;
  const popup = new maplibregl.Popup({ closeButton: false })
    .setLngLat(coordinates).setDOMContent(node).addTo(map);
  node.querySelector('button').addEventListener('click', () => {
    addWaypoint({
      name: props.ident,
      lat: props.lat, lon: props.lon,
      kind: 'airport',
      elevation_ft: props.elevation_ft,
      label: props.label,
    });
    popup.remove();
  });
}

// --- route editing ------------------------------------------------------

function addWaypoint(waypoint) {
  // A new point asks the planner what its leg should do. In user-driven mode
  // the pilot has to answer before the plan will build, which is the intended
  // prompt rather than an error to avoid.
  route.push({ segment_type: 'automatic', generated: false, id: newId(), ...waypoint });
  onRouteChanged();
  fitRoute();
}

/** Frame the whole route. Called when waypoints are added or removed, but
 *  deliberately not while dragging a marker -- the map moving under the
 *  cursor mid-drag is disorienting. */
function fitRoute() {
  if (route.length < 2) return;
  const bounds = route.reduce(
    (b, w) => b.extend([w.lon, w.lat]),
    new maplibregl.LngLatBounds([route[0].lon, route[0].lat], [route[0].lon, route[0].lat]),
  );
  map.fitBounds(bounds, { padding: 80, maxZoom: 10, duration: 600 });
}

/** Insert a clicked point into whichever leg it lies closest to. */
function insertWaypointAt(lngLat) {
  let bestLeg = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < route.length - 1; i += 1) {
    const d = distanceToSegment(lngLat, route[i], route[i + 1]);
    if (d < bestDistance) { bestDistance = d; bestLeg = i; }
  }
  const inserted = {
    name: nextWaypointName(),
    lat: +lngLat.lat.toFixed(5),
    lon: +lngLat.lng.toFixed(5),
    kind: 'waypoint',
    elevation_ft: null,
    id: newId(),
  };
  splitLegAt(bestLeg + 1, inserted);
  route.splice(bestLeg + 1, 0, inserted);
  onRouteChanged();
}

/** Carry what was said about a leg across a point dropped into it.
 *
 *  The leg the new point splits is named by the pilot waypoints either side
 *  of it. Its events go to whichever half they lie on, and its edits are
 *  copied to both halves: a wind typed on the whole leg is still the wind on
 *  each part of it. */
function splitLegAt(slot, inserted) {
  const before = route.slice(0, slot).reverse().find((w) => !w.generated);
  const after = route.slice(slot).find((w) => !w.generated);
  if (!before || !after) return;
  const at = projectOntoLeg({ lng: inserted.lon, lat: inserted.lat }, before, after).t;
  const events = after.events || [];
  inserted.events = events.filter(
    (e) => projectOntoLeg({ lng: e.lon, lat: e.lat }, before, after).t < at);
  after.events = events.filter((e) => !inserted.events.includes(e));
  copyEdits(`${before.id}>${after.id}`, [`${before.id}>${inserted.id}`, `${inserted.id}>${after.id}`]);
}

/** The next free `WP<n>`. Searched, not counted: `WP${route.length}` handed
 *  out a name already in use after any insert-and-delete, and names have to
 *  be unique -- see `renameWaypoint`. */
function nextWaypointName() {
  const taken = new Set(route.map((w) => w.name.toUpperCase()));
  let n = 1;
  while (taken.has(`WP${n}`)) n += 1;
  return `WP${n}`;
}

/** Whether a name can be given to this waypoint.
 *
 *  Unique, because the route is matched by name in two places: the pilot's
 *  own fields are carried across a re-resolve by name (`adoptResolvedRoute`),
 *  and the field weather is cached by name (`fieldWx`). Two points called the
 *  same thing would silently swap their crossing altitudes. And not one of the
 *  planner's names: `TOC`, `TOD`, `TOC2`... are what it calls the points it
 *  inserts, and a pilot's "TOD" would be stripped before every re-plan.
 */
function isNameAvailable(name, waypoint) {
  if (!name) return false;
  if (/^[TB]O[CD]\d*$/i.test(name)) return false;
  const upper = name.toUpperCase();
  return !route.some((w) => w !== waypoint && w.name.toUpperCase() === upper);
}

/** Planar point-to-segment distance, with longitude scaled by latitude.
 *  Only used to pick which leg a click belongs to, so exactness is wasted. */
function distanceToSegment(point, a, b) {
  const k = Math.cos((a.lat * Math.PI) / 180);
  const px = point.lng * k, py = point.lat;
  const ax = a.lon * k, ay = a.lat;
  const bx = b.lon * k, by = b.lat;
  const dx = bx - ax, dy = by - ay;
  const lengthSq = dx * dx + dy * dy;
  const t = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSq));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function removeWaypoint(index) {
  const gone = route[index];
  if (!gone.generated) {
    // The two legs either side become one. Its events are both sets, and its
    // edits are the first leg's where both had one.
    const before = route.slice(0, index).reverse().find((w) => !w.generated);
    const after = route.slice(index + 1).find((w) => !w.generated);
    if (before && after) {
      after.events = [...(gone.events || []), ...(after.events || [])];
      const merged = `${before.id}>${after.id}`;
      copyEdits(`${gone.id}>${after.id}`, [merged]);
      copyEdits(`${before.id}>${gone.id}`, [merged]);
    }
  }
  route.splice(index, 1);
  onRouteChanged();
  fitRoute();
}

function moveWaypoint(from, to) {
  const [item] = route.splice(from, 1);
  route.splice(to, 0, item);
  onRouteChanged();
}

function onRouteChanged() {
  // Edits are filed under the leg they were typed on, not a row, so nothing
  // here has to chase them: a leg that still exists keeps its edits, and one
  // that is gone simply stops being sent. See `segmentEdits`.
  ensureIds();
  // The derived pressure and density altitudes are
  // indexed by route position, and the next plan is what re-earns them.
  if (fieldAir.length !== route.length) fieldAir = [];
  // The forecasts were fetched over the points that just moved, and for the
  // times the old route reached them. Both claims are now false.
  clearForecasts();
  // And the briefing was for a corridor that is now somewhere else. A stale
  // NOTAM list beside a changed route is worse than none: it reads as having
  // been checked.
  notamReport = null;
  renderNotams();
  clearConsistency();
  renderWaypointList();
  renderMarkers();
  renderRouteLine();
  requestPlan();
}

/** A leg's vertical profile changed, not where it goes: an event was added,
 *  moved or retargeted. The forecasts and the NOTAM corridor still stand. */
function onProfileChanged() {
  clearConsistency();
  renderWaypointList();
  renderEventMarkers();
  requestPlan();
}

// --- rendering ----------------------------------------------------------

/** Below this, the ground is close enough that the air at it is a question.
 *  Above it the FD levels are pressure altitudes already and carry their own
 *  temperatures, so a field altimeter setting has nothing left to say. */
const LOW_ALTITUDE_FT = 3000;

/** Whether this waypoint is a field.
 *
 *  The database stamps an airport with its size -- `large_airport`,
 *  `medium_airport`, `small_airport` -- and that is what a search result
 *  carries into the route. Only the map popup stamps a bare "airport". Every
 *  test that asked for the bare word therefore missed every airport added
 *  from the search box: Get weather found no airports on the route, an
 *  airport's published position was offered as editable lat/lon boxes, and an
 *  intermediate field had no "stop" checkbox. Size is not a different sort of
 *  thing, so it is asked about in one place.
 *
 *  `server/main.py:_is_airport` is the same predicate on the other side.
 */
function isAirport(waypoint) {
  return waypoint.kind === 'airport' || String(waypoint.kind || '').endsWith('_airport');
}

/** Text for an `innerHTML` template. A waypoint's name is the pilot's own
 *  since it became renameable, and a name typed as `<b>` must print as one. */
function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/** Whether this waypoint is somewhere the air *at the ground* matters.
 *
 *  Every airport, because one may become a stop and all of them report; the
 *  departure and destination whatever they are, because a strip with no
 *  database entry still has to be got out of; and any point crossed low,
 *  where an elevation and a temperature are what make the density altitude
 *  beside it mean anything.
 */
function isFieldPoint(waypoint, index) {
  if (waypoint.generated) return false;
  if (index === 0 || index === route.length - 1) return true;
  if (isAirport(waypoint) || waypoint.is_landing) return true;
  if (waypoint.elevation_ft != null) return true;
  return waypoint.altitude_ft != null && waypoint.altitude_ft < LOW_ALTITUDE_FT;
}

/** Field elevation, altimeter setting and temperature, plus what they work
 *  out to.
 *
 *  The elevation is editable even for an airport whose figure came from the
 *  database: the database is a stopgap, a private strip is not in it at all,
 *  and a takeoff distance read at the wrong field elevation is wrong in the
 *  direction that matters. Blank means "use the published figure" for an
 *  airport and "unknown" for anything else.
 *
 *  Altimeter setting and temperature both fall back to the route-wide figures
 *  when left blank, which the placeholders say. Departure and destination are
 *  often an hour and a different airmass apart, so each gets its own.
 *
 *  The surface wind is entered true, the way the METAR gives it; the engine
 *  converts it to magnetic to line it up with the runway designators. It picks
 *  the runway end, corrects the takeoff and landing distances, and its
 *  crosswind is checked against the maximum demonstrated. Blank leaves the
 *  go/no-go on the no-wind book figures and says so beside them.
 */
function fieldBlock(waypoint, index) {
  if (!isFieldPoint(waypoint, index)) return '';
  return `<div class="coords field">` +
    `<label title="Field or ground elevation, in feet MSL">Elev` +
      `<input class="elev-in" type="number" step="1" min="-1500" max="15000" ` +
        `placeholder="${isAirport(waypoint) ? 'published' : 'unknown'}" ` +
        `value="${waypoint.elevation_ft ?? ''}"></label>` +
    `<label title="Field altimeter setting, off this station's METAR or ATIS">` +
      `Alt&nbsp;set<input class="qnh" type="number" step="0.01" min="27" max="32" ` +
        `placeholder="route" value="${waypoint.altimeter_inhg ?? ''}"></label>` +
    `<label title="Field temperature">OAT °C` +
      `<input class="oat" type="number" step="1" min="-40" max="55" ` +
        `placeholder="ISA" value="${waypoint.oat_c ?? ''}"></label>` +
    `</div>` +
    `<div class="coords field">` +
    `<label title="Surface wind direction, degrees TRUE as the METAR reports ` +
      `it. Blank leaves the runway distances at their no-wind figures">Wind` +
      `<input class="wind-from" type="number" step="10" min="0" max="360" ` +
        `placeholder="—" value="${waypoint.wind_from_deg ?? ''}"></label>` +
    `<label title="Surface wind speed, knots. 0 is calm">kt` +
      `<input class="wind-kt" type="number" step="1" min="0" max="99" ` +
        `placeholder="—" value="${waypoint.wind_speed_kt ?? ''}"></label>` +
    `<label title="Peak gust, knots. Used for the crosswind and for a ` +
      `tailwind, ignored where it would only flatter the numbers">Gust` +
      `<input class="wind-gust" type="number" step="1" min="0" max="99" ` +
        `placeholder="—" value="${waypoint.gust_kt ?? ''}"></label>` +
    `</div>` +
    fieldWxText(waypoint) +
    // Filled in from the last plan, so the pressure and density altitude the
    // engine computed are the ones shown -- there is no second implementation
    // of the atmosphere in the browser to drift from it.
    `<div class="field-air" data-index="${index}">${fieldAirText(index)}</div>`;
}

/** Where this field's numbers came from, after a "Get weather".
 *
 *  Only for the boxes that still hold what was fetched. A pilot who has since
 *  typed over the temperature is not flying a METAR temperature, and a line
 *  that kept saying so would be the kind of quiet lie this whole panel exists
 *  to avoid. An entry with nothing left to claim disappears.
 */
function fieldWxText(waypoint) {
  const wx = fieldWx.get(waypoint.name);
  if (!wx) return '';
  if (wx.error) {
    return `<div class="field-wx none">No weather for ${waypoint.name}: ` +
      `${wx.error}</div>`;
  }
  const still = (field) => waypoint[field] === wx.wrote[field];
  const named = [
    ['wind_from_deg', 'wind'],
    ['oat_c', 'OAT'],
    ['altimeter_inhg', 'alt set'],
  ].filter(([field]) => field in wx.wrote && still(field));
  if (!named.length) return '';
  const from = named
    .map(([field, label]) => `${label} ${wx.sources[wxSourceKey(field)] || '?'}`)
    .join(' · ');
  const notes = wx.notes.length ? ` title="${wx.notes.join('; ')}"` : '';
  return `<div class="field-wx"${notes}>${wx.station} ` +
    `${zulu(wx.valid_time)} — ${from}</div>`;
}

/** The `sources` key a filled-in box was resolved under.
 *
 *  Direction and speed arrive as one decision in the engine -- a wind is a
 *  vector or it is nothing -- so both boxes answer to the one source. */
function wxSourceKey(field) {
  return field === 'wind_from_deg' || field === 'wind_speed_kt' ? 'wind' : field;
}

/** An ISO instant as the four-digit Zulu time a report is stamped with. */
function zulu(iso) {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(when.getUTCHours())}${pad(when.getUTCMinutes())}Z`;
}

/** The derived line under one waypoint's field inputs, or empty. */
function fieldAirText(index) {
  const air = fieldAir[index];
  if (!air || air.pressure_altitude_ft == null) return '';
  const ft = (v) => `${Math.round(v).toLocaleString()} ft`;
  return `PA ${ft(air.pressure_altitude_ft)} · ` +
    `<strong>DA ${ft(air.density_altitude_ft)}</strong> ` +
    `<span class="from">at ${air.altimeter_inhg.toFixed(2)} inHg, ` +
    `${Math.round(air.oat_c)} °C</span>`;
}

/** Repaint the derived lines in place after a plan.
 *
 *  In place rather than by re-rendering the list: the pilot is often still
 *  typing in another box when the debounced plan comes back, and rebuilding
 *  the sidebar under them would take the caret with it.
 */
function renderFieldAir() {
  for (const node of document.querySelectorAll('.field-air')) {
    node.innerHTML = fieldAirText(+node.dataset.index);
  }
}

// --- leg events ---------------------------------------------------------
//
// "Start the climb here" and "be level by here", pinned to a place on the leg
// arriving at a waypoint. Hybrid mode only: the planner pins one end of the
// altitude change there and lets the wind decide where the other end falls.

// [button, chip text before the altitude, tooltip, short map label]
const EVENT_LABELS = {
  start: ['Start climb/descent here', 'from here climb/descend to',
    'Hold altitude until here, then climb or descend to the altitude in the box. '
    + 'BOC (or TOD) is pinned here; where you level off floats with the wind.', 'start →'],
  complete: ['Level by here', 'be at',
    'Be at the altitude in the box by here. TOC (or BOD) is pinned here; '
    + 'where the climb begins floats with the wind.', 'by'],
};

/** The leg a waypoint's events are on starts at the pilot waypoint before it. */
function legStartFor(waypoint) {
  const index = route.indexOf(waypoint);
  return route.slice(0, index).reverse().find((w) => !w.generated) || null;
}

/** Events in the order they are flown along their leg. */
function orderedEvents(waypoint) {
  const from = legStartFor(waypoint);
  const events = waypoint.events || [];
  if (!from) return events;
  return [...events].sort((a, b) =>
    projectOntoLeg({ lng: a.lon, lat: a.lat }, from, waypoint).t
    - projectOntoLeg({ lng: b.lon, lat: b.lat }, from, waypoint).t);
}

function eventBlock(waypoint, index) {
  if (planningMode !== 'hybrid' || waypoint.generated || index === 0) return '';
  const from = legStartFor(waypoint);
  if (!from) return '';
  const picking = eventPick?.waypoint === waypoint;
  const rows = orderedEvents(waypoint).map((e) => {
    const [, short, title] = EVENT_LABELS[e.kind];
    const i = waypoint.events.indexOf(e);
    return `<span class="event ev-${e.kind}" title="${title}">${short}` +
      `<input class="ev-alt" data-i="${i}" type="number" step="500" min="0" max="17999" ` +
      `aria-label="Altitude to ${e.kind === 'start' ? 'climb or descend to' : 'be at'}" ` +
      `value="${Math.round(e.target_altitude_ft)}"> ft` +
      (e.kind === 'complete' ? ' by here' : '') +
      `<button class="ev-del" data-i="${i}" type="button" title="Remove this event">×</button></span>`;
  }).join('');
  const add = Object.entries(EVENT_LABELS).map(([kind, [label, , title]]) =>
    `<button class="ev-add${picking && eventPick.kind === kind ? ' picking' : ''}" ` +
    `data-kind="${kind}" type="button" title="Click the map on this leg: ${title}">+ ${label}</button>`).join('');
  return `<div class="events"><span class="events-title">` +
    `Leg from ${escapeHtml(from.name)}</span>${rows}<span class="ev-adds">${add}</span></div>`;
}

function bindEventBlock(li, waypoint) {
  li.querySelectorAll('.ev-add').forEach((button) => button.addEventListener('click', () => {
    const kind = button.dataset.kind;
    const again = eventPick?.waypoint === waypoint && eventPick.kind === kind;
    setEventPick(again ? null : { waypoint, kind });
  }));
  li.querySelectorAll('.ev-del').forEach((button) => button.addEventListener('click', () => {
    waypoint.events.splice(+button.dataset.i, 1);
    onProfileChanged();
  }));
  li.querySelectorAll('.ev-alt').forEach((input) => input.addEventListener('change', () => {
    const value = Number(input.value);
    if (input.value.trim() === '' || !Number.isFinite(value) || value < 0 || value > 17999) {
      input.value = Math.round(waypoint.events[+input.dataset.i].target_altitude_ft);
      return;
    }
    waypoint.events[+input.dataset.i].target_altitude_ft = value;
    onProfileChanged();
  }));
}

/** Waiting for a click on the map to place an event, or null. */
let eventPick = null;

function setEventPick(pick) {
  eventPick = pick;
  document.body.classList.toggle('picking-event', !!pick);
  if (pick) {
    const from = legStartFor(pick.waypoint);
    setStatus(`Click the leg ${from?.name} → ${pick.waypoint.name} where to `
      + `${pick.kind === 'start' ? 'start the climb or descent' : 'be level'}. Esc cancels.`);
  } else {
    setStatus('Ready — offline');
  }
  renderWaypointList();
}

/** Put the pending event where the pilot clicked, snapped onto its leg. */
function placeEvent(lngLat) {
  const { waypoint, kind } = eventPick;
  const from = legStartFor(waypoint);
  setEventPick(null);
  if (!from) return;
  const at = projectOntoLeg(lngLat, from, waypoint);
  // The altitude the leg is headed for: its end's crossing altitude where it
  // has one, the cruise altitude otherwise. The box beside it is for changing.
  waypoint.events = [...(waypoint.events || []), {
    kind, lat: at.lat, lon: at.lon,
    target_altitude_ft: waypoint.altitude_ft ?? +$('altitude').value,
  }];
  onProfileChanged();
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && eventPick) setEventPick(null);
});

let eventMarkers = [];

/** The pilot's events on the map. Draggable along their leg. */
function renderEventMarkers() {
  eventMarkers.forEach((m) => m.remove());
  eventMarkers = [];
  if (planningMode !== 'hybrid') return;
  for (const waypoint of route) {
    const from = legStartFor(waypoint);
    if (waypoint.generated || !from) continue;
    for (const e of waypoint.events || []) {
      const element = document.createElement('div');
      element.className = `marker event ev-${e.kind}`;
      const label = document.createElement('div');
      label.className = 'marker-label dim';
      label.textContent = `${EVENT_LABELS[e.kind][3]} ${Math.round(e.target_altitude_ft)}`;
      element.appendChild(label);
      const marker = new maplibregl.Marker({ element, draggable: true })
        .setLngLat([e.lon, e.lat]).addTo(map);
      marker.on('dragend', () => {
        const at = projectOntoLeg(marker.getLngLat(), from, waypoint);
        e.lat = at.lat;
        e.lon = at.lon;
        onProfileChanged();
      });
      eventMarkers.push(marker);
    }
  }
}

/** Whether the flight lands at this point: the destination, or a stop. */
function landsHere(waypoint, index) {
  return index === route.length - 1 || (isAirport(waypoint) && !!waypoint.is_landing);
}

function renderWaypointList() {
  const list = $('waypoints');
  list.innerHTML = '';
  route.forEach((waypoint, index) => {
    const li = document.createElement('li');
    li.dataset.index = index;
    // Colours the identifier magenta for a VFR checkpoint, matching the map.
    li.classList.add(`kind-${waypoint.kind}`);
    // A TOC/TOD the planner inserted, dimmed because it is not the pilot's.
    if (waypoint.generated) li.classList.add('generated');
    // Only an intermediate airport can be a stop: the first and last points
    // are always a departure and a destination, and are landed at anyway.
    const canLand = isAirport(waypoint)
      && index > 0 && index < route.length - 1;
    const landing = canLand
      ? `<label class="landing" title="Land here — adds a descent in and a climb out">` +
        `<input type="checkbox" ${waypoint.is_landing ? 'checked' : ''}> stop</label>`
      : '';
    // What the leg *arriving* here does. The first waypoint is departed from,
    // never arrived at, so it has no segment of its own.
    const type = waypoint.segment_type || 'automatic';
    // In user-driven mode "automatic" is not a choice, but it is still what a
    // freshly added point holds. It gets a placeholder rather than nothing, so
    // the box shows an unanswered question instead of silently reading
    // "climb" while the route is really undeclared and will refuse to plan.
    const segment = index === 0
      ? ''
      : `<select class="segment seg-${type}" ` +
        `title="What the leg arriving here does about altitude">` +
        (planningMode === 'hybrid'
          ? `<option value="automatic" ${type === 'automatic' ? 'selected' : ''}>auto</option>`
          : (type === 'automatic'
            ? `<option value="automatic" selected disabled>choose…</option>`
            : '')) +
        SEGMENT_TYPES.map((t) =>
          `<option value="${t}" ${type === t ? 'selected' : ''}>${t}</option>`).join('') +
        `</select>`;
    const role = waypoint.generated ? `<span class="role">${waypoint.name}</span>` : '';
    // An airport's position comes from the database and is not ours to move.
    // A point dropped on the map is arbitrary, so it can be typed exactly --
    // off a chart, or to place a fix on an airway intersection.
    const freeform = !isAirport(waypoint);
    const coords = freeform
      ? `<div class="coords">` +
        `<label>Lat<input class="lat" type="number" step="0.0001" ` +
          `min="-90" max="90" value="${waypoint.lat}"></label>` +
        `<label>Lon<input class="lon" type="number" step="0.0001" ` +
          `min="-180" max="180" value="${waypoint.lon}"></label>` +
        `<label title="Cross this point at this altitude">Cross` +
          `<input class="alt" type="number" step="500" min="0" max="17999" ` +
            `placeholder="cruise" value="${waypoint.altitude_ft ?? ''}"></label>` +
        `</div>`
      : `<div class="coords readonly">` +
          `<span>${waypoint.lat.toFixed(4)}, ${waypoint.lon.toFixed(4)}</span>` +
          // An airport can be crossed at an altitude too. On the field the
          // flight lands at -- the last one, or a stop -- it is where the
          // descent ends; blank ends it at the field's pattern altitude.
          (index > 0
            ? `<label title="${landsHere(waypoint, index)
              ? 'Where the descent into this field ends. Blank: the pattern altitude (field + 1,000 ft, or its published TPA).'
              : 'Cross this field at this altitude'}">Cross` +
              `<input class="alt" type="number" step="100" min="0" max="17999" ` +
              `placeholder="${landsHere(waypoint, index) ? 'TPA' : 'cruise'}" ` +
              `value="${waypoint.altitude_ft ?? ''}"></label>`
            : '') +
        `</div>`;
    // A dropped point's name is the pilot's to change -- it is what the navlog
    // and the map label print. An airport's is its identifier: the server
    // looks its runways up by it (`WaypointIn.ident`), so it stays as it is.
    const name = freeform && !waypoint.generated
      ? `<input class="wp-name" type="text" maxlength="10" spellcheck="false" ` +
        `title="Rename this waypoint" value="${escapeHtml(waypoint.name)}">`
      : `<span class="name">${escapeHtml(waypoint.name)}</span>`;
    li.innerHTML =
      `<div class="wp-row">` +
        `<span class="drag" title="Drag to reorder">⠿</span>` +
        `<span class="seq">${index + 1}</span>` +
        name +
        segment +
        landing +
        `<button class="remove" type="button" title="Remove">×</button>` +
      `</div>` + coords + eventBlock(waypoint, index) + fieldBlock(waypoint, index);
    li.querySelector('.remove').addEventListener('click', () => removeWaypoint(index));
    bindEventBlock(li, waypoint);

    const nameBox = li.querySelector('input.wp-name');
    if (nameBox) {
      // Ten characters because the text export lays its FROM and TO columns
      // out that wide (`format_navlog`); a longer name would push the row.
      nameBox.addEventListener('change', () => {
        const value = nameBox.value.trim();
        if (!isNameAvailable(value, waypoint)) {
          nameBox.value = waypoint.name;
          return;
        }
        if (value === waypoint.name) return;
        waypoint.name = value;
        onRouteChanged();
      });
      nameBox.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') nameBox.blur();
        if (e.key === 'Escape') { nameBox.value = waypoint.name; nameBox.blur(); }
      });
    }

    {
      const bind = (selector, apply) => {
        const input = li.querySelector(selector);
        if (!input) return;
        input.addEventListener('change', () => {
          const raw = input.value.trim();
          const value = raw === '' ? null : Number(raw);
          if (value !== null && !Number.isFinite(value)) return;
          if (!apply(value)) return;
          onRouteChanged();
        });
      };
      bind('.lat', (v) => {
        if (v === null || v < -90 || v > 90) return false;
        waypoint.lat = v; return true;
      });
      bind('.lon', (v) => {
        if (v === null || v < -180 || v > 180) return false;
        waypoint.lon = v; return true;
      });
      // Blank means "no constraint" -- fly the route's cruise altitude.
      bind('.alt', (v) => { waypoint.altitude_ft = v; return true; });
      // Below sea level is real (Death Valley), above 15000 ft is not a field
      // this airplane is leaving from.
      bind('.elev-in', (v) => {
        if (v !== null && (v < -1500 || v > 15000)) return false;
        waypoint.elevation_ft = v;
        return true;
      });
      // Blank means "use the route's weather" for this field.
      bind('.qnh', (v) => {
        if (v !== null && (v < 27 || v > 32)) return false;
        waypoint.altimeter_inhg = v;
        // The setting the pilot just read off the departure ATIS is almost
        // always the one to fly the whole route on. Offered only while the
        // route field is still at standard, so it never overwrites a figure
        // somebody chose.
        if (v !== null && index === 0 && +$('altimeter').value === 29.92) {
          $('altimeter').value = v;
        }
        return true;
      });
      bind('.oat', (v) => { waypoint.oat_c = v; return true; });
      // True degrees, as reported. 360 and 0 are the same wind; the engine
      // wraps it either way.
      bind('.wind-from', (v) => {
        if (v !== null && (v < 0 || v > 360)) return false;
        waypoint.wind_from_deg = v;
        return true;
      });
      bind('.wind-kt', (v) => {
        if (v !== null && (v < 0 || v > 99)) return false;
        waypoint.wind_speed_kt = v;
        return true;
      });
      // A "gust" at or below the steady wind is not one; the engine drops it
      // rather than refusing the plan over it.
      bind('.wind-gust', (v) => {
        if (v !== null && (v < 0 || v > 99)) return false;
        waypoint.gust_kt = v;
        return true;
      });
    }
    const picker = li.querySelector('select.segment');
    if (picker) {
      picker.addEventListener('change', () => {
        waypoint.segment_type = picker.value;
        // Choosing a type for a planner-inserted point makes it the pilot's,
        // so the next resolve keeps it instead of discarding and re-deriving.
        if (waypoint.generated) {
          waypoint.generated = false;
          waypoint.kind = 'waypoint';
        }
        // The profile changes shape here, but not necessarily under every
        // row: the next plan drops the edits whose legs actually moved.
        onRouteChanged();
      });
    }
    const box = li.querySelector('.landing input');
    if (box) {
      box.addEventListener('change', () => {
        waypoint.is_landing = box.checked;
        // A stop changes which rows exist; the next plan drops the edits
        // whose legs went with them and keeps the rest.
        onRouteChanged();
      });
    }
    bindDragToReorder(li);
    list.appendChild(li);
  });
  $('route-hint').hidden = route.length >= 2;
}

/*  Reordering runs on pointer events rather than HTML5 drag-and-drop: Safari
 *  on iPadOS never fires dragstart from a finger, so the bar could only be
 *  reordered with a mouse. Pointer events cover both, and the gesture starts
 *  on the ⠿ handle alone -- a drag anywhere else in the row would fight the
 *  text fields and the list's own scrolling.
 *
 *  The row is not moved while the finger is down. Only the insertion line
 *  follows the pointer, and the route is spliced once on release: a live
 *  reorder would renumber the rows mid-gesture and re-plan on every frame. */
let dragState = null;

/*  Which slot the pointer is over -- 0..route.length, counting the gaps
 *  between rows, not the rows. Rows differ in height (a landing row carries
 *  a weather block), so each is measured rather than assumed. */
function dropSlotAt(list, clientY) {
  const rows = [...list.children];
  for (let i = 0; i < rows.length; i++) {
    const rect = rows[i].getBoundingClientRect();
    if (clientY < rect.top + rect.height / 2) return i;
  }
  return rows.length;
}

function showDropLine(list, slot) {
  const rows = [...list.children];
  rows.forEach((row) => row.classList.remove('drop-above', 'drop-below'));
  if (slot === null) return;
  if (slot < rows.length) rows[slot].classList.add('drop-above');
  else if (rows.length) rows[rows.length - 1].classList.add('drop-below');
}

function endReorder(commit) {
  if (!dragState) return;
  const { li, list, handle, pointerId, slot } = dragState;
  dragState = null;
  li.classList.remove('dragging');
  showDropLine(list, null);
  if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
  if (!commit || slot === null) return;
  const from = +li.dataset.index;
  // The slot counts gaps in the list as it stands, with the dragged row still
  // in it; once that row is spliced out every gap below it shifts up one.
  const to = slot > from ? slot - 1 : slot;
  if (to !== from) moveWaypoint(from, to);
}

function bindDragToReorder(li) {
  const handle = li.querySelector('.drag');
  if (!handle) return;
  handle.addEventListener('pointerdown', (e) => {
    // Left button or a finger; a right-click must not start a reorder.
    if (e.button !== 0) return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    dragState = {
      li, list: li.parentElement, handle, pointerId: e.pointerId,
      startY: e.clientY, slot: null, active: false,
    };
  });
  handle.addEventListener('pointermove', (e) => {
    if (!dragState || dragState.pointerId !== e.pointerId) return;
    // A few pixels of slop, so a tap that trembles is still a tap.
    if (!dragState.active) {
      if (Math.abs(e.clientY - dragState.startY) < 5) return;
      dragState.active = true;
      li.classList.add('dragging');
    }
    dragState.slot = dropSlotAt(dragState.list, e.clientY);
    showDropLine(dragState.list, dragState.slot);
  });
  handle.addEventListener('pointerup', (e) => {
    if (!dragState || dragState.pointerId !== e.pointerId) return;
    endReorder(dragState.active);
  });
  // A cancel is the browser taking the gesture over -- a scroll it decided
  // was one, or a palm. The route is left as it was.
  handle.addEventListener('pointercancel', () => endReorder(false));
}

function renderMarkers() {
  markers.forEach((m) => m.remove());
  markers = route.map((waypoint, index) => {
    const element = document.createElement('div');
    const isEndpoint = index === 0 || index === route.length - 1;
    element.className = `marker${isEndpoint ? ' endpoint' : ''}`;
    const label = document.createElement('div');
    label.className = 'marker-label';
    label.textContent = waypoint.name;
    element.appendChild(label);

    const marker = new maplibregl.Marker({ element, draggable: true })
      .setLngLat([waypoint.lon, waypoint.lat])
      .addTo(map);
    marker.on('dragend', () => {
      const position = marker.getLngLat();
      route[index].lat = +position.lat.toFixed(5);
      route[index].lon = +position.lng.toFixed(5);
      // Dragging an airport off its published position makes it a plain
      // waypoint; keeping the identifier would be a lie.
      if (isAirport(route[index]) && index !== 0 && index !== route.length - 1) {
        route[index].kind = 'waypoint';
      }
      onRouteChanged();
    });
    return marker;
  });
  renderEventMarkers();
}

function renderRouteLine() {
  if (!map.getSource('route')) return;
  const features = [];

  // Draw from the plan's legs when there is one. The plan contains top of
  // climb and top of descent, which the raw route does not, so its leg list
  // is longer -- indexing the route by leg number would paint the whole line
  // the colour of the first phase.
  if (lastPlan?.ok) {
    for (const leg of lastPlan.legs) {
      // Taxi and pattern rows sit at a single point; drawing them would put a
      // zero-length line under the airport marker.
      if (!leg.covers_ground) continue;
      features.push({
        type: 'Feature',
        properties: { phase: leg.phase },
        geometry: {
          type: 'LineString',
          coordinates: [[leg.from_lon, leg.from_lat], [leg.to_lon, leg.to_lat]],
        },
      });
    }
  } else {
    for (let i = 0; i < route.length - 1; i += 1) {
      features.push({
        type: 'Feature',
        properties: { phase: 'cruise' },
        geometry: {
          type: 'LineString',
          coordinates: [[route[i].lon, route[i].lat], [route[i + 1].lon, route[i + 1].lat]],
        },
      });
    }
  }
  map.getSource('route').setData({ type: 'FeatureCollection', features });
  renderPhaseMarkers();
}

/** Small markers at top of climb and top of descent. */
let phaseMarkers = [];
function renderPhaseMarkers() {
  phaseMarkers.forEach((m) => m.remove());
  phaseMarkers = [];
  if (!lastPlan?.ok) return;
  for (const leg of lastPlan.legs) {
    // Match the role rather than the name, so a numbered TOC2 on a
    // multi-stop day gets a marker too, and a charted point nominated as the
    // top of climb is marked under its own name.
    if (!leg.end_role) continue;
    const element = document.createElement('div');
    element.className = 'marker phase';
    const label = document.createElement('div');
    label.className = 'marker-label dim';
    label.textContent = legLabel(leg.to, leg.end_role);
    element.appendChild(label);
    phaseMarkers.push(
      new maplibregl.Marker({ element }).setLngLat([leg.to_lon, leg.to_lat]).addTo(map),
    );
  }
}

// --- manual edits, filed under the leg they were typed on ---------------
//
// An edit belongs to the leg the pilot drew -- "<from id>><to id>" -- and
// optionally to one phase of it, never to a row. The rows a leg is cut into
// move every time the wind moves a top of climb; the leg does not. So a wind
// typed on SUNOL → VPCLB is still on that leg however many rows the next plan
// cuts it into, and still there when the mission is saved and loaded again
// tomorrow. Which is the point: the log is something to keep editing, not
// something to rebuild whenever the weather changes.
//
// Keyed `${segment_key}|${phase}`, with an empty phase for the whole leg.

let segmentEdits = new Map();

const editKey = (segment, phase) => `${segment}|${phase || ''}`;

/** What a typed wind covers: the whole leg, or only the row's phase of it.
 *  Temperature, pressure altitude and airspeed are always the phase's own --
 *  the climb and the cruise are flown in different air at different speeds. */
let windScope = 'leg';

function scopeOf(field) {
  if (field === 'wind_from_deg' || field === 'wind_speed_kt') return windScope;
  return field === 'altitude_ft' ? 'leg' : 'phase';
}

function updateEdit(key, change) {
  const entry = { ...(segmentEdits.get(key) || {}) };
  change(entry);
  if (Object.keys(entry).length) segmentEdits.set(key, entry);
  else segmentEdits.delete(key);
}

/** Copy one leg's edits onto others, where they have none of their own. Used
 *  when a waypoint splits a leg in two or its removal joins two into one. */
function copyEdits(fromLeg, toLegs) {
  for (const [key, fields] of [...segmentEdits.entries()]) {
    const [leg, phase] = key.split('|');
    if (leg !== fromLeg) continue;
    for (const target of toLegs) {
      updateEdit(editKey(target, phase), (entry) => {
        for (const [field, value] of Object.entries(fields)) entry[field] ??= value;
      });
    }
  }
}

function setSegmentEdit(leg, field, value) {
  const phaseKey = editKey(leg.segment_key, leg.phase);
  const legKey = editKey(leg.segment_key, null);
  if (value === null) {
    // Blank clears whichever entry is supplying the number: the phase's own
    // first, since that is the one on show.
    const key = segmentEdits.get(phaseKey)?.[field] != null ? phaseKey : legKey;
    updateEdit(key, (entry) => { delete entry[field]; });
  } else if (scopeOf(field) === 'leg') {
    updateEdit(legKey, (entry) => { entry[field] = value; });
    // A leg-wide value typed over a phase's own replaces it; otherwise the
    // phase's would go on hiding the number just typed.
    updateEdit(phaseKey, (entry) => { delete entry[field]; });
  } else {
    updateEdit(phaseKey, (entry) => { entry[field] = value; });
  }
  $('reset-edits').hidden = segmentEdits.size === 0;
  requestPlan();
}

/** Every edit on a leg of the current route. Edits on legs that no longer
 *  exist -- the route was reordered -- are kept but not sent: put the points
 *  back and they apply again. */
function segmentOverridesPayload() {
  const live = new Set(pilotLegs().map((leg) => leg.key));
  return [...segmentEdits.entries()]
    .filter(([key]) => live.has(key.split('|')[0]))
    .map(([key, fields]) => {
      const [segment_key, phase] = key.split('|');
      return { segment_key, phase: phase || null, ...fields };
    });
}

function clearLegEdits(segment) {
  for (const key of [...segmentEdits.keys()]) {
    if (key.split('|')[0] === segment) segmentEdits.delete(key);
  }
  $('reset-edits').hidden = segmentEdits.size === 0;
  requestPlan();
}

// Committing an edit replans, which rebuilds the whole table and destroys the
// input the pilot was typing in. Remember where the cursor was so it can be
// put back, or tabbing from wind direction to wind speed silently drops focus
// and the next keystrokes go nowhere.
let focusedCell = null;
const FIELD_CLASS = {
  wind_from_deg: 'wind-dir',
  wind_speed_kt: 'wind-speed',
  tas_kt: 'tas',
  altitude_ft: 'alt',
  oat_c: 'oat',
  pressure_altitude_ft: 'pa',
};

/** A waypoint name with its top-of-climb/descent role beside it.
 *
 *  `TOC` alone where the planner invented the point, `KWVI (TOC)` where the
 *  pilot nominated a charted one -- the name on the sectional is what they
 *  will be looking for, so it is never replaced.
 */
function legLabel(name, role) {
  // A planner point -- TOC2, BOC -- already says what it is.
  return !role || name === role || /^[TB]O[CD]\d*$/.test(name) ? name : `${name} (${role})`;
}

function restoreFocus() {
  if (!focusedCell) return;
  const { row, field, start, end } = focusedCell;
  const tr = document.querySelector(`#navlog tbody tr[data-row="${row}"]`);
  const input = tr && tr.querySelector(`td.${FIELD_CLASS[field]} input`);
  if (!input) { focusedCell = null; return; }
  input.focus();
  try { input.setSelectionRange(start, end); } catch { /* number inputs vary */ }
}

function clearOverrides() {
  if (segmentEdits.size === 0) return;
  segmentEdits = new Map();
  const reset = $('reset-edits');
  if (reset) reset.hidden = true;
}

/** Drop the consistency verdict. Called whenever the route changes, so a
 *  clean bill of health never outlives the plan it was given for. */
function clearConsistency() {
  if (consistencyReport === null) return;
  consistencyReport = null;
  renderConsistency(null);
}

/** Turn a table cell into a number the pilot can type over.
 *
 *  What is typed is filed under the row's leg (and phase), so the title says
 *  which leg -- the pilot is editing that stretch of route, not this row. */
function makeEditable(cell, { leg, row, field, value, format, title }) {
  cell.classList.add('editable');
  const scope = scopeOf(field) === 'leg' ? 'the whole leg' : `the ${leg.phase} on it`;
  cell.title = `${title}\nApplies to ${legName(leg.segment_key)}, ${scope}.`;
  if (leg.overridden.includes(field)) cell.classList.add('edited');

  const input = document.createElement('input');
  input.type = 'number';
  input.value = format(value);
  input.setAttribute('aria-label', title);
  cell.textContent = '';
  cell.appendChild(input);

  const commit = () => {
    const raw = input.value.trim();
    const next = raw === '' ? null : Number(raw);
    if (next !== null && !Number.isFinite(next)) return;
    setSegmentEdit(leg, field, next);
  };
  input.addEventListener('focus', () => {
    focusedCell = { row, field, start: 0, end: input.value.length };
  });
  input.addEventListener('select', () => {
    if (focusedCell) {
      focusedCell.start = input.selectionStart ?? 0;
      focusedCell.end = input.selectionEnd ?? 0;
    }
  });
  input.addEventListener('change', commit);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { focusedCell = null; input.blur(); }
    if (event.key === 'Escape') { focusedCell = null; setSegmentEdit(leg, field, null); }
  });
}

// --- planning -----------------------------------------------------------

let planTimer = null;
function requestPlan() {
  clearTimeout(planTimer);
  // Debounced so dragging a marker does not fire a request per frame.
  planTimer = setTimeout(runPlan, 180);
}

/** The plan request for the current route and form. Shared by the plan call
 *  and the consistency check, so the two can never describe different flights. */
function planBody() {
  return {
    waypoints: route.map((w) => ({
      name: w.name, lat: w.lat, lon: w.lon, kind: w.kind,
      elevation_ft: w.elevation_ft, ident: w.name,
      is_landing: !!w.is_landing,
      altitude_ft: w.altitude_ft ?? null,
      segment_type: w.segment_type || 'automatic',
      generated: !!w.generated,
      id: w.generated ? null : (w.id ?? null),
      events: w.generated ? [] : (w.events || []),
      altimeter_inhg: w.altimeter_inhg ?? null,
      oat_c: w.oat_c ?? null,
      wind_from_deg: w.wind_from_deg ?? null,
      wind_speed_kt: w.wind_speed_kt ?? null,
      gust_kt: w.gust_kt ?? null,
      // The rest of the field's report, for the VFR half of the go/no-go.
      // `sky_reported` travels with them: without it the engine cannot tell a
      // reported clear sky from a source that never looked.
      visibility_sm: w.visibility_sm ?? null,
      ceiling_ft_agl: w.ceiling_ft_agl ?? null,
      ceiling_cover: w.ceiling_cover ?? '',
      sky_reported: !!w.sky_reported,
      weather_reported: !!w.weather_reported,
    })),
    planning_mode: planningMode,
    segment_overrides: segmentOverridesPayload(),
    forecasts,
    // What turns a cumulative ETE into a clock time, and so what decides
    // which forecast hour each leg is read at.
    off_blocks: offBlocksUtc()?.toISOString() ?? null,
    runway_margin: +$('runway-margin').value / 100,
    fuel_margin: +$('fuel-margin').value / 100,
    cruise_altitude_ft: +$('altitude').value,
    cruise_rpm: +$('rpm').value,
    weight_lb: +$('weight').value,
    fuel_on_board_gal: +$('fuel').value,
    altimeter_inhg: +$('altimeter').value,
    isa_deviation_c: +$('isadev').value,
    night: $('night').checked,
    // No route-wide wind: it is typed on the leg it applies to, and travels
    // to the engine in `segment_overrides`.
  };
}

async function runPlan() {
  if (route.length < 2) {
    lastPlan = null;
    adoptFieldAir(null);
    renderNavlog(null);
    return;
  }
  setStatus('Planning…');
  ensureIds();
  lastPlan = await api.plan(planBody());
  // Before adopting: adoption may re-render the sidebar, and the derived
  // lines should already be right when it does.
  adoptFieldAir(lastPlan?.ok ? lastPlan.resolved_waypoints : null);
  if (lastPlan?.ok) adoptResolvedRoute(lastPlan.resolved_waypoints);
  renderFieldAir();
  renderNavlog(lastPlan);
  renderRouteLine();
  refreshCsvDrawer();
  // renderNavlog owns the status when it refuses, so it can say why.
  if (lastPlan?.ok) setStatus('Ready — offline');
}

/** Keep the pressure and density altitude the engine computed at each field.
 *
 *  Indexed by position in the resolved route, which is the route the sidebar
 *  is about to show. A refused plan clears them rather than leaving yesterday's
 *  density altitude under today's temperature.
 */
function adoptFieldAir(resolved) {
  fieldAir = Array.isArray(resolved)
    ? resolved.map((w) => ({
      altimeter_inhg: w.field_altimeter_inhg,
      oat_c: w.field_oat_c,
      pressure_altitude_ft: w.field_pressure_altitude_ft,
      density_altitude_ft: w.field_density_altitude_ft,
    }))
    : [];
}

/** Take the planner's TOC/TOD points into the route the pilot is editing.
 *
 *  Only rewrites the list when it actually differs, because rewriting it
 *  re-renders the sidebar and would fight with whatever the pilot is typing.
 *  Safe to do on every plan: the engine strips its own generated points before
 *  re-resolving, so this cannot compound.
 */
function adoptResolvedRoute(resolved) {
  if (!Array.isArray(resolved) || !resolved.length) return;
  // A planner-owned point is compared on where it *is*, not just what it is
  // called: a top of climb moves along the route whenever the plan changes
  // under it -- a new cruise altitude, or a wind typed on the climb row -- and
  // the shape of the list does not change with it. Without this the sidebar
  // keeps showing the previous TOC's position and crossing altitude while the
  // map draws the current one, which is two answers to the same question.
  //
  // Only for generated points. The pilot's own are left alone so that
  // re-rendering never fights with whatever they are typing into.
  const settled = (w, existing) =>
    Math.abs(w.lat - existing.lat) < 1e-6
    && Math.abs(w.lon - existing.lon) < 1e-6
    && Math.abs((w.altitude_ft ?? 0) - (existing.altitude_ft ?? 0)) < 1;
  const same = resolved.length === route.length && resolved.every((w, i) =>
    w.name === route[i].name
    && !!w.generated === !!route[i].generated
    && (w.segment_type || 'automatic') === (route[i].segment_type || 'automatic')
    && (!w.generated || settled(w, route[i])));
  if (same) return;
  route = resolved.map((w, i) => ({
    // Keep anything of the pilot's the engine does not round-trip. Matched by
    // name, which `isNameAvailable` keeps unique across the route.
    ...(w.generated ? {} : route.find((r) => r.name === w.name) || {}),
    // The id the pilot's point was sent with, which is what its leg's events
    // and edits are filed under. A planner point has none.
    ...(w.generated ? {} : { id: w.id }),
    name: w.name, lat: w.lat, lon: w.lon, kind: w.kind,
    elevation_ft: w.elevation_ft,
    is_landing: !!w.is_landing,
    altitude_ft: w.altitude_ft,
    segment_type: w.segment_type,
    generated: !!w.generated,
    altimeter_inhg: w.altimeter_inhg,
    oat_c: w.oat_c,
    wind_from_deg: w.wind_from_deg,
    wind_speed_kt: w.wind_speed_kt,
    gust_kt: w.gust_kt,
  }));
  renderWaypointList();
  renderMarkers();
}

/** The row that heads one drawn leg's rows: its name, its events, its edits. */
function legHeading(segment) {
  const tr = document.createElement('tr');
  tr.className = 'leg-head';
  const edits = [...segmentEdits.entries()]
    .filter(([key]) => key.split('|')[0] === segment)
    .map(([key, fields]) => {
      const phase = key.split('|')[1] || 'whole leg';
      const parts = [];
      if (fields.wind_from_deg != null || fields.wind_speed_kt != null) {
        parts.push(`wind ${fields.wind_from_deg ?? '—'}/${fields.wind_speed_kt ?? '—'}`);
      }
      for (const [field, label] of [['tas_kt', 'TAS'], ['oat_c', 'OAT'],
        ['pressure_altitude_ft', 'PA'], ['altitude_ft', 'alt']]) {
        if (fields[field] != null) parts.push(`${label} ${fields[field]}`);
      }
      return `${phase}: ${parts.join(', ')}`;
    });
  const leg = pilotLegs().find((l) => l.key === segment);
  const events = leg ? (leg.to.events || []).length : 0;
  tr.innerHTML = `<td colspan="24"><span class="leg-name">${escapeHtml(legName(segment))}</span>` +
    (events ? `<span class="leg-meta">${events} event${events === 1 ? '' : 's'}</span>` : '') +
    (edits.length
      ? `<span class="leg-meta edited">${escapeHtml(edits.join(' · '))}</span>` +
        `<button type="button" class="leg-reset" title="Clear the edits on this leg and go back to the forecast">Reset leg</button>`
      : '') +
    `</td>`;
  tr.querySelector('.leg-reset')?.addEventListener('click', () => clearLegEdits(segment));
  return tr;
}

/** Minutes left after a row, to the end of the flight, pattern included:
 *  it counts down to 0 the way the fuel Rem column does. */
function timeRemText(leg) {
  const total = lastPlan?.ok ? lastPlan.totals.time_min : null;
  if (total == null || leg.cumulative_ete_min == null) return '';
  return Math.max(0, total - leg.cumulative_ete_min).toFixed(1);
}

function renderNavlog(plan) {
  const body = document.querySelector('#navlog tbody');
  const foot = document.querySelector('#navlog tfoot');
  const empty = $('navlog-empty');
  body.innerHTML = '';
  foot.innerHTML = '';
  $('summary-block').hidden = true;
  $('warnings').innerHTML = '';
  // Cleared up front, not on the success path, so that every early return
  // below leaves it hidden. A stale "GO" beside a route that would not plan is
  // the single most dangerous thing this screen could show -- and a weather
  // list left standing beside no navlog claims to describe rows that are not
  // there.
  renderChecklist(null);

  if (!plan) {
    empty.hidden = false;
    empty.className = 'hint';
    empty.textContent = 'Add at least two waypoints to build a navigation log.';
    return;
  }
  if (!plan.ok) {
    // A refusal to plan is a result, not a blank screen. Styled as an alert
    // and echoed into the status bar, because "Ready" beside an empty table
    // reads as "nothing to show" rather than "here is what is wrong".
    empty.hidden = false;
    empty.className = 'alert';
    empty.textContent = plan.error;
    setStatus('Cannot plan this route');
    return;
  }
  empty.className = 'hint';
  empty.hidden = true;

  const round = (v, d = 0) => v.toFixed(d);
  let heading = null;
  plan.legs.forEach((leg, row) => {
    // Rows are grouped under the leg the pilot drew them on: that is what an
    // edit is typed against, and what its events belong to.
    if (leg.covers_ground && leg.segment_key && leg.segment_key !== heading) {
      heading = leg.segment_key;
      body.appendChild(legHeading(leg.segment_key));
    }
    const tr = document.createElement('tr');
    tr.dataset.row = row;
    if (leg.covers_ground && leg.segment_key) tr.classList.add('in-leg');
    if (leg.overridden.length) tr.classList.add('row-edited');
    // A row whose performance was extrapolated. Marked on the row so the log
    // itself shows where the totals stop being the book's, not only the
    // checklist underneath it.
    if (leg.extrapolated) {
      tr.classList.add('row-extrapolated');
      tr.title = (leg.off_chart || []).map((e) => e.detail).join('; ');
    }

    // Taxi and the traffic pattern cost time and fuel without covering
    // ground, so they get a row -- the fuel column adds up to the total --
    // but every navigation column on it is a dash rather than a number that
    // looks flyable. Nothing on the row is editable either.
    if (!leg.covers_ground) {
      tr.classList.add('row-ground');
      tr.innerHTML =
        `<td>${leg.from}</td><td></td><td></td><td></td>` +
        `<td class="phase-${leg.phase}">${leg.phase}</td>` +
        `<td class="num">${round(leg.altitude_ft)}</td>` +
        // The air is real even where the row goes nowhere: this is the field's
        // density altitude, the one the takeoff distance was read at.
        `<td class="num">${round(leg.oat_c)}</td>` +
        `<td class="num">${round(leg.pressure_altitude_ft)}</td>` +
        `<td class="num">${round(leg.density_altitude_ft)}</td>` +
        // Course through ground speed, plus distance: eleven columns of nothing.
        '<td class="num">—</td>'.repeat(11) +
        `<td class="num">${round(leg.ete_min, 1)}</td>` +
        `<td class="num">${timeRemText(leg)}</td>` +
        `<td class="num">${round(leg.fuel_gal, 1)}</td>` +
        `<td class="num">${round(leg.fuel_remaining_gal, 1)}</td>`;
      body.appendChild(tr);
      return;
    }

    tr.innerHTML =
      `<td>${legLabel(leg.from, leg.start_role)}</td>` +
      `<td>${legLabel(leg.to, leg.end_role)}</td>` +
      // Read-only here: a navlog row may start at a synthetic TOC or TOD,
      // which has no independent existence to move. Arbitrary waypoints are
      // edited in the route list, where they actually live.
      `<td class="num coord">${leg.from_lat.toFixed(4)}</td>` +
      `<td class="num coord">${leg.from_lon.toFixed(4)}</td>` +
      `<td class="phase-${leg.phase}">${leg.phase}</td>` +
      `<td class="num alt"></td>` +
      `<td class="num oat"></td>` +
      `<td class="num pa"></td>` +
      // Derived, never typed: whatever pressure altitude and temperature the
      // row ends up with, this follows from the pair of them.
      `<td class="num">${round(leg.density_altitude_ft)}</td>` +
      // The full chain a pilot works through: true course, crab into wind for
      // true heading, then variation for the magnetic heading actually flown.
      `<td class="num">${round(leg.true_course_deg).padStart(3, '0')}</td>` +
      `<td class="num">${leg.wind_correction_angle_deg >= 0 ? '+' : ''}` +
        `${round(leg.wind_correction_angle_deg, 1)}</td>` +
      `<td class="num">${round(leg.true_heading_deg).padStart(3, '0')}</td>` +
      `<td class="num">${leg.variation_deg >= 0 ? '+' : ''}${round(leg.variation_deg, 1)}</td>` +
      `<td class="num">${round(leg.magnetic_heading_deg).padStart(3, '0')}</td>` +
      `<td class="num wind-dir"></td>` +
      `<td class="num wind-speed"></td>` +
      `<td class="num">${leg.cas_kt == null ? '—' : round(leg.cas_kt)}</td>` +
      `<td class="num tas"></td>` +
      `<td class="num">${round(leg.ground_speed_kt)}</td>` +
      `<td class="num">${round(leg.distance_nm, 1)}</td>` +
      `<td class="num">${round(leg.ete_min, 1)}</td>` +
      `<td class="num">${timeRemText(leg)}</td>` +
      `<td class="num">${round(leg.fuel_gal, 1)}</td>` +
      `<td class="num">${round(leg.fuel_remaining_gal, 1)}</td>`;

    // The cells the pilot can overwrite. Everything else on the row is
    // derived from them and recomputes on the next plan.
    makeEditable(tr.querySelector('.wind-dir'), {
      leg, row, field: 'wind_from_deg', value: leg.wind_from_deg,
      format: (v) => Math.round(v),
      title: 'Wind direction on this leg, degrees true. Blank for calm.',
    });
    makeEditable(tr.querySelector('.wind-speed'), {
      leg, row, field: 'wind_speed_kt', value: leg.wind_speed_kt,
      format: (v) => Math.round(v),
      title: 'Wind speed on this leg, knots. Blank for calm.',
    });
    // The altitude this leg *ends* at, which carries forward to every row
    // after it -- editing the top of a climb re-flies the rest of the plan.
    // User-driven only: in hybrid the planner places the altitudes, and the
    // pilot pins them with events on the leg instead.
    if (planningMode === 'manual') {
      makeEditable(tr.querySelector('.alt'), {
        leg, row, field: 'altitude_ft',
        value: leg.exit_altitude_ft ?? leg.altitude_ft,
        format: (v) => Math.round(v),
        title: 'Altitude at the end of this leg. Blank to compute it again.',
      });
    } else {
      const cell = tr.querySelector('.alt');
      cell.textContent = round(leg.exit_altitude_ft ?? leg.altitude_ft);
      cell.title = 'Placed by the planner. Pin a climb or descent with an event '
        + 'on the leg in the route list, or a crossing altitude on a waypoint.';
    }
    // Temperature is the odd one out: it describes the air, not the row, so
    // it joins the field temperatures in the flight's profile and lapses into
    // the legs above and below rather than stopping at this one.
    makeEditable(tr.querySelector('.oat'), {
      leg, row, field: 'oat_c', value: leg.oat_c,
      format: (v) => Math.round(v),
      title: 'Outside air temperature at this altitude, °C — the FD forecast '
        + 'figure for this part of the route. It lapses into the rows above '
        + 'and below. Blank to go back to the standard lapse.',
    });
    // Pressure altitude, which the route's altimeter setting normally decides.
    // Typing one says the air over this leg does not match that setting; the
    // density altitude beside it and the charts this row reads follow.
    makeEditable(tr.querySelector('.pa'), {
      leg, row, field: 'pressure_altitude_ft', value: leg.pressure_altitude_ft,
      format: (v) => Math.round(v),
      title: 'Pressure altitude of this row\'s air, ft — what the altimeter '
        + 'reads with 29.92 set. Density altitude and this row\'s chart '
        + 'readings follow from it. Blank to take it from the altimeter '
        + 'setting again.',
    });
    makeEditable(tr.querySelector('.tas'), {
      leg, row, field: 'tas_kt', value: leg.tas_kt,
      format: (v) => Math.round(v),
      title: 'True airspeed, knots. Blank to go back to the POH figure.',
    });
    body.appendChild(tr);
  });

  const t = plan.totals;
  const tr = document.createElement('tr');
  tr.innerHTML =
    `<td colspan="19">Total</td>` +
    `<td class="num">${round(t.distance_nm, 1)}</td>` +
    `<td class="num">${round(t.time_min, 1)}</td>` +
    `<td class="num"></td>` +
    `<td class="num">${round(t.fuel_gal, 1)}</td>` +
    `<td class="num">${round(t.fuel_remaining_gal, 1)}</td>`;
  foot.appendChild(tr);

  renderSummary(plan);
  renderChecklist(plan.checklist);
  restoreFocus();
}

/** The weather list, one row per leg the pilot drew.
 *
 *  Shows the comparison that decided each leg rather than just its result: a
 *  pilot who can see that the departure end costs 5.4 gal and the arrival end
 *  3.6 can tell at a glance whether the choice was close or obvious, and
 *  whether the day is worth waiting out.
 */
// --- go / no-go ---------------------------------------------------------

function renderChecklist(checklist) {
  const block = $('checklist-block');
  const body = $('checklist');
  body.innerHTML = '';
  if (!checklist) { block.hidden = true; return; }
  block.hidden = false;

  const verdict = $('verdict');
  // Three states. "GO — EXTRAPOLATED" is still a go: every check passed, but
  // some of the numbers behind it came from air the POH does not publish for
  // that operating point, and a word that hid that would be the wrong word.
  const extrapolated = checklist.is_go && checklist.all_from_the_book === false;
  verdict.textContent = checklist.verdict
    || (checklist.is_go ? 'GO' : 'NO GO');
  verdict.className =
    `verdict ${!checklist.is_go ? 'nogo' : extrapolated ? 'extrapolated' : 'go'}`;

  const ft = (v) => (v == null ? '—' : `${Math.round(v)}`);
  const mark = (p) => (p === true ? 'ok' : p === false ? 'short' : 'unknown');
  const label = (p) => (p === true ? 'OK' : p === false ? 'SHORT' : '?');
  // The field as a whole, where "SHORT" would be a lie: a field can fail on
  // its sky with 10,000 ft of runway under it.
  const fieldLabel = (p) => (p === true ? 'OK' : p === false ? 'NO GO' : '?');
  // Signed, because the sign is the whole story: -6 on the runway you were
  // going to use is a longer roll, not a shorter one.
  const kt = (v) => (v == null ? '—' : `${v > 0 ? '+' : ''}${Math.round(v)}`);
  // What the sky is doing, in the words the report used.
  const skyText = (wx) => {
    if (wx.obscured) return `sky obscured (${wx.ceiling_cover})`;
    if (wx.ceiling_ft_agl != null) {
      return `${wx.ceiling_cover || 'ceiling'} ${ft(wx.ceiling_ft_agl)} ft AGL`;
    }
    return wx.sky_reported ? 'no ceiling' : 'sky not reported';
  };
  // What the facts above do not already say. The full reason is spelled out
  // in the blocker list below -- repeating it here would print the ceiling
  // twice in one line -- so all this adds is the number the ceiling missed,
  // and the note for a sky nobody could judge.
  const weatherDetail = (wx) => {
    if (wx.passes === null) return ` — ${wx.summary}`;
    const short = wx.required_ceiling_ft_agl != null && wx.ceiling_ft_agl != null
      && wx.ceiling_ft_agl < wx.required_ceiling_ft_agl;
    return short ? ` — needs ${ft(wx.required_ceiling_ft_agl)} ft` : '';
  };
  const windText = (c) => {
    if (c.wind_speed_kt == null) return 'wind not given';
    if (c.wind_speed_kt === 0) return 'wind calm';
    const gust = c.gust_kt == null ? '' : `G${Math.round(c.gust_kt)}`;
    return `Wind ${String(Math.round(c.wind_from_deg)).padStart(3, '0')}°M at ` +
      `${Math.round(c.wind_speed_kt)}${gust} kt`;
  };

  for (const check of checklist.airports) {
    const section = document.createElement('div');
    section.className = 'check';
    // The conditions the numbers were computed at, shown beside them: a
    // surprising distance is nearly always a surprising density altitude.
    section.innerHTML =
      `<div class="check-head">` +
        `<span class="check-title">${check.airport} ${check.operation}</span>` +
        `<span class="pill ${mark(check.passes)}">${fieldLabel(check.passes)}</span>` +
      `</div>` +
      `<div class="check-conditions">` +
        `Field ${ft(check.elevation_ft)} ft · OAT ${Math.round(check.oat_c)} °C · ` +
        `Pressure alt ${ft(check.pressure_altitude_ft)} ft · ` +
        `<strong>Density alt ${ft(check.density_altitude_ft)} ft</strong> · ` +
        `${ft(check.weight_lb)} lb · margin ${Math.round(check.margin * 100)}% · ` +
        `<strong>${windText(check)}</strong>` +
      `</div>`;

    // The sky, on its own line. It is a gate in its own right -- the longest
    // runway on the field is no use under an obscuration -- so it gets a
    // verdict of its own rather than being folded in among the conditions.
    if (check.weather) {
      const wx = check.weather;
      const row = document.createElement('div');
      row.className = `check-weather ${mark(wx.passes)}`;
      row.innerHTML =
        `<span class="pill ${mark(wx.passes)}">` +
          `${wx.passes === true ? 'VFR' : wx.passes === false ? 'NOT VFR' : '?'}` +
        `</span> ${skyText(wx)} · ` +
        `${wx.visibility_sm == null ? 'visibility not reported'
          : `${wx.visibility_sm} sm visibility`} · ` +
        `pattern ${ft(wx.pattern_altitude_agl_ft)} ft AGL` +
        weatherDetail(wx);
      section.appendChild(row);
    }

    const table = document.createElement('table');
    table.className = 'runways';
    table.innerHTML =
      `<thead><tr><th>Runway</th><th>Surface</th>` +
      `<th title="The end with the better headwind — the one these numbers ` +
        `are for">Use</th>` +
      `<th class="num" title="Headwind on that end; negative is a tailwind">` +
        `Head</th>` +
      `<th class="num" title="Crosswind, at the gust where there is one">` +
        `Cross</th>` +
      `<th class="num">Length</th><th class="num">Roll</th>` +
      `<th class="num" title="Book distance over a 50 ft obstacle">Book 50</th>` +
      `<th class="num" title="Book distance plus your margin">Required</th>` +
      `<th class="num">Spare</th><th></th></tr></thead><tbody></tbody>`;
    const tbody = table.querySelector('tbody');
    for (const runway of check.runways) {
      const tr = document.createElement('tr');
      tr.className = mark(runway.passes);
      tr.innerHTML =
        `<td>${runway.runway || '—'}</td>` +
        `<td>${runway.surface || '—'}${runway.dry_grass_applied ? ' (grass)' : ''}</td>` +
        `<td>${runway.end_used || '—'}</td>` +
        `<td class="num">${kt(runway.headwind_kt)}</td>` +
        `<td class="num${runway.crosswind_exceeds_demonstrated ? ' over' : ''}">` +
          `${ft(runway.crosswind_kt)}` +
          // A side on a crosswind that rounds to nothing is noise: straight
          // down the runway has no left or right about it.
          `${runway.crosswind_kt >= 0.5
            ? (runway.crosswind_from_right ? ' R' : ' L') : ''}` +
        `</td>` +
        `<td class="num">${ft(runway.runway_available_ft)}</td>` +
        `<td class="num">${ft(runway.ground_roll_ft)}</td>` +
        `<td class="num">${ft(runway.over_50ft_ft)}</td>` +
        `<td class="num">${ft(runway.required_ft)}</td>` +
        `<td class="num">${ft(runway.spare_ft)}</td>` +
        `<td><span class="pill ${mark(runway.passes)}">` +
        // A blank chart cell, a crosswind past the demonstrated maximum and a
        // runway that is simply too short all read as no-go, but they are not
        // the same problem and the pilot acts on each differently.
        `${runway.outside_envelope ? 'NO DATA'
          : runway.crosswind_exceeds_demonstrated ? 'XWIND'
            : label(runway.passes)}</span></td>`;
      tbody.appendChild(tr);
      if (runway.note) {
        const note = document.createElement('tr');
        note.className = 'note';
        note.innerHTML = `<td colspan="11">${runway.note}</td>`;
        tbody.appendChild(note);
      }
    }
    section.appendChild(table);
    body.appendChild(section);
  }

  const fuel = checklist.fuel;
  const fuelSection = document.createElement('div');
  fuelSection.className = 'check';
  const spareMin = fuel.spare_minutes == null
    ? '' : ` (${Math.round(fuel.spare_minutes)} min)`;
  fuelSection.innerHTML =
    `<div class="check-head">` +
      `<span class="check-title">Fuel reserve</span>` +
      `<span class="pill ${mark(fuel.passes)}">${label(fuel.passes)}</span>` +
    `</div>` +
    `<div class="check-conditions">` +
      `${fuel.reserve_minutes} minutes ${fuel.night ? 'night' : 'day'} VFR ` +
      `(FAR 91.151) · margin ${Math.round(fuel.margin * 100)}%` +
    `</div>` +
    `<dl class="fuel-figures">` +
      `<div><dt>On board</dt><dd>${fuel.fuel_on_board_gal.toFixed(1)} gal</dd></div>` +
      `<div><dt>Burn</dt><dd>${fuel.burn_gal.toFixed(1)} gal</dd></div>` +
      `<div><dt>Lands with</dt><dd>${fuel.landing_with_gal.toFixed(1)} gal</dd></div>` +
      `<div><dt>Reserve</dt><dd>${fuel.reserve_required_gal.toFixed(1)} gal</dd></div>` +
      `<div><dt>Required</dt><dd>${fuel.required_with_margin_gal.toFixed(1)} gal</dd></div>` +
      `<div><dt>Spare</dt><dd>${fuel.spare_gal.toFixed(1)} gal${spareMin}</dd></div>` +
    `</dl>`;
  body.appendChild(fuelSection);

  for (const reason of checklist.blockers) {
    const div = document.createElement('div');
    div.className = 'alert';
    div.textContent = `NO GO — ${reason}`;
    body.appendChild(div);
  }
  for (const reason of checklist.unknowns) {
    const div = document.createElement('div');
    div.className = 'alert unknown';
    div.textContent = `Unknown — ${reason}`;
    body.appendChild(div);
  }
  // Last, under the verdict they qualify: these are not failures, they are the
  // reason the word above may read GO with a qualifier on it. The ones that
  // can read optimistic carry the caution colour; a conservative substitution
  // errs long and is a note, not a warning.
  for (const entry of checklist.extrapolations || []) {
    const div = document.createElement('div');
    div.className =
      `alert ${entry.conservative ? 'extrapolated' : 'extrapolated-optimistic'}`;
    div.textContent = `Extrapolated — ${entry.where}: ${entry.detail}`;
    body.appendChild(div);
  }
}

function renderSummary(plan) {
  const t = plan.totals;
  const hours = Math.floor(t.time_min / 60);
  const minutes = Math.round(t.time_min % 60);
  $('summary').innerHTML = [
    ['Distance', `${t.distance_nm.toFixed(1)} nm`],
    ['Time en route', hours ? `${hours}h ${minutes}m` : `${minutes} min`],
    ['Fuel burn', `${t.fuel_gal.toFixed(1)} gal`],
    ['Landing with', `${t.fuel_remaining_gal.toFixed(1)} gal`],
    ['Reserve needed', `${t.reserve_required_gal.toFixed(1)} gal`],
    ['Fuel legal', t.legal_on_fuel ? 'yes' : 'NO'],
  ].map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');

  const warnings = $('warnings');
  // Where the wind on every row came from. A forecast column and a number the
  // pilot read off a chart look identical in the table, and they are not the
  // same thing to be flying on -- so the table says which it is holding.
  if (forecastWx) {
    const div = document.createElement('div');
    div.className = 'alert model';
    const missed = forecastWx.of - forecastWx.points;
    const passes = plan.weather_passes;
    div.textContent =
      `Winds and temperatures aloft: model forecast over ${forecastWx.points} `
      + `point${forecastWx.points === 1 ? '' : 's'}, `
      + `${forecastWx.hours} hour${forecastWx.hours === 1 ? '' : 's'} each. `
      + `Each leg is planned in whichever of its two ends costs more fuel`
      + (passes ? `; settled in ${passes} pass${passes === 1 ? '' : 'es'}` : '')
      + (missed ? `; ${missed} point(s) unavailable` : '')
      + '. A forecast, not an observation — and no substitute for a briefing.';
    warnings.appendChild(div);
  }
  // What the day's weather changed, against the plan the mission was saved
  // with. Said first: it is why the pilot loaded the mission.
  const changed = sinceSaved(plan);
  if (changed) {
    const div = document.createElement('div');
    div.className = 'hint since-saved';
    div.textContent = changed;
    warnings.appendChild(div);
  }
  for (const message of plan.warnings) {
    const div = document.createElement('div');
    div.className = 'alert';
    div.textContent = message;
    warnings.appendChild(div);
  }
  $('summary-block').hidden = false;
}

// --- airport search box -------------------------------------------------

let searchTimer = null;
$('search').addEventListener('input', (event) => {
  clearTimeout(searchTimer);
  const query = event.target.value.trim();
  if (query.length < 2) { $('results').hidden = true; return; }
  searchTimer = setTimeout(async () => {
    const found = await api.searchAirports(query);
    renderSearchResults(found);
  }, 150);
});

$('search').addEventListener('keydown', (event) => {
  if (event.key === 'Escape') { $('results').hidden = true; }
  if (event.key === 'Enter') {
    const first = $('results').querySelector('button');
    if (first) first.click();
  }
});

function renderSearchResults(found) {
  const box = $('results');
  box.innerHTML = '';
  if (!found.length) { box.hidden = true; return; }
  for (const airport of found) {
    const button = document.createElement('button');
    button.type = 'button';
    const runway = airport.longest_runway_ft
      ? ` · ${Math.round(airport.longest_runway_ft)} ft` : '';
    // The one box searches airports and published VFR checkpoints together,
    // and a checkpoint has no elevation -- `Math.round(null)` is 0, which
    // would read as a field at sea level rather than as a bridge.
    const elevation = airport.elevation_ft == null
      ? '' : ` · ${Math.round(airport.elevation_ft)} ft`;
    button.innerHTML =
      `<span class="ident">${airport.ident}</span> ${airport.name}` +
      `<span class="meta">${airport.municipality ?? ''} ${airport.region ?? ''}` +
      `${elevation}${runway}</span>`;
    button.addEventListener('click', () => {
      addWaypoint({
        name: airport.ident, lat: airport.lat, lon: airport.lon,
        // Whatever the result actually is. Stamping every search result as an
        // airport made a VFR checkpoint claim to be a field: it offered a
        // "land here" box, and Get weather asked a weather service about a
        // bridge and got a 404 back.
        kind: airport.kind || 'airport',
        elevation_ft: airport.elevation_ft, label: airport.label,
      });
      $('search').value = '';
      box.hidden = true;
      // Only recentre on the first waypoint. After that `addWaypoint` has
      // already framed the whole route, and flying to the newest point would
      // undo that -- which is exactly what it used to do.
      if (route.length === 1) {
        map.flyTo({ center: [airport.lon, airport.lat], zoom: Math.max(map.getZoom(), 8) });
      }
    });
    box.appendChild(button);
  }
  box.hidden = false;
}

document.addEventListener('click', (event) => {
  if (!event.target.closest('.search')) $('results').hidden = true;
});

// --- form and misc ------------------------------------------------------

for (const id of ['altitude', 'rpm', 'weight', 'fuel', 'altimeter', 'isadev',
                  'night']) {
  $(id).addEventListener('change', requestPlan);
}

$('reset-edits').addEventListener('click', () => {
  clearOverrides();
  requestPlan();
});

for (const id of ['runway-margin', 'fuel-margin']) {
  $(id).addEventListener('change', requestPlan);
}

/** Fill every airport on the route from its current surface weather.
 *
 *  One button for the whole route rather than one per field: the fields are
 *  fetched together anyway, and a pilot who wants the weather wants all of it
 *  before deciding to go. The three requests per station are the server's
 *  problem -- it caches them -- so this is one round trip per airport.
 *
 *  Only what the report actually gives is written. A station with no
 *  temperature leaves the temperature box alone rather than blanking it, and
 *  the values are rounded to what a pilot would have typed off the ATIS: the
 *  half-degree thrown away is under 60 ft of density altitude, well inside the
 *  margin the checklist already carries.
 */
/** The off-blocks time as a UTC instant, or null when the box is empty.
 *
 *  `datetime-local` has no timezone, and the label says Z, so the typed value
 *  is read as UTC rather than as the laptop's local time. A pilot planning in
 *  California for a Zulu departure should not have the browser silently shift
 *  it by seven hours.
 */
function offBlocksUtc() {
  const typed = $('off-blocks').value;
  if (!typed) return null;
  const at = new Date(`${typed}:00Z`);
  return Number.isNaN(at.getTime()) ? null : at;
}

/** When each airport on the route is actually reached, from the log's ETEs.
 *
 *  Keyed by waypoint name. The departure field is the off-blocks time itself;
 *  every later field is off-blocks plus the cumulative ETE of the leg that
 *  *arrives* there, which is what the navlog already computes for the row.
 *
 *  Arrival rather than departure time at an intermediate stop: it is the
 *  landing that the runway check is about, and the pattern and taxi that
 *  follow are ten minutes against weather published by the hour.
 *
 *  Empty when there is no off-blocks time or no plan to read ETEs from, which
 *  is the caller's signal to fetch current weather instead.
 */
function plannedArrivals() {
  const start = offBlocksUtc();
  const times = new Map();
  if (!start || !lastPlan?.ok || !Array.isArray(lastPlan.legs)) return times;
  for (const leg of lastPlan.legs) {
    if (!leg.to || times.has(leg.to)) continue;  // first arrival wins
    const minutes = leg.cumulative_ete_min;
    if (minutes == null) continue;
    times.set(leg.to, new Date(start.getTime() + minutes * 60000));
  }
  // The departure field is reached at the off-blocks time, not after the
  // taxi leg that happens to name it as a destination.
  if (route.length) times.set(route[0].name, start);
  return times;
}

async function getWeather() {
  const button = $('get-weather');
  // Airports only: a fix or a private strip has no station to ask about, and
  // the ident is what the route sends as the waypoint's name.
  const fields = route.filter((w, i) =>
    !w.generated && isAirport(w) && isFieldPoint(w, i));
  if (!fields.length) {
    flashButton(button, 'No airports on the route');
    return;
  }

  button.disabled = true;

  // Pass one. The ETEs that place each field in time come from the navlog
  // itself, so there has to be a log before there can be arrival times. Only
  // when a time was asked for -- fetching the current weather needs no plan.
  if (offBlocksUtc() && !lastPlan?.ok) {
    button.textContent = 'Planning…';
    await runPlan();
  }
  const arrivals = plannedArrivals();

  button.textContent = 'Fetching…';
  let reports;
  try {
    reports = await Promise.all(fields.map(
      (w) => api.surface(w.name, arrivals.get(w.name)?.toISOString())));
  } catch {
    // The one thing worse than no weather is a button that stays greyed out
    // because the network went away mid-fetch.
    button.disabled = false;
    flashButton(button, 'Weather unavailable');
    return;
  }
  button.disabled = false;

  let filled = 0;
  const failed = [];
  fields.forEach((waypoint, n) => {
    const report = reports[n];
    if (!report || report.error) {
      fieldWx.set(waypoint.name, { error: (report && report.error) || 'no answer' });
      failed.push(waypoint.name);
      return;
    }
    const wrote = {};
    if (report.altimeter_inhg != null) {
      wrote.altimeter_inhg = Math.round(report.altimeter_inhg * 100) / 100;
    }
    if (report.oat_c != null) wrote.oat_c = Math.round(report.oat_c);
    // Direction and speed together or not at all -- half a wind is not one,
    // and a direction with no speed would read as a calm from the north.
    if (report.wind_from_deg != null && report.wind_speed_kt != null) {
      wrote.wind_from_deg = Math.round(report.wind_from_deg);
      wrote.wind_speed_kt = Math.round(report.wind_speed_kt);
      // A report with no gust in it says there is no gust. Keeping the last
      // one would hold a peak that has since dropped against the crosswind
      // limit, so the wind is replaced whole. What arrives is already known
      // to belong to this wind -- `weather.resolve_surface` drops a gust
      // that came from a weaker source than the wind it would attach to.
      wrote.gust_kt = report.gust_kt == null ? null : Math.round(report.gust_kt);
    }
    Object.assign(waypoint, wrote);
    // The sky and the visibility go straight onto the waypoint rather than
    // through `wrote`: they fill no box on the form, so they are not part of
    // what "filled 2 fields" counts or of what the source line describes.
    //
    // Replaced whole, on the same reasoning as the gust. A report that no
    // longer mentions an overcast is saying the overcast has gone, and
    // keeping the last one would hold a ceiling against a field that has
    // cleared. A report with no sky group at all leaves `sky_reported` false,
    // which the go/no-go reads as unknown rather than as clear.
    Object.assign(waypoint, {
      visibility_sm: report.visibility_sm ?? null,
      ceiling_ft_agl: report.ceiling_ft_agl ?? null,
      ceiling_cover: report.ceiling_cover ?? '',
      sky_reported: !!report.sky_reported,
      // A report came back, whatever was in it. A model-only answer has no
      // cloud and no visibility in it, and the go/no-go has to be able to
      // tell that from a field nobody asked about.
      weather_reported: true,
    });
    // The departure setting is the one to fly the route on, on the same
    // reasoning as typing it by hand -- and only while the route field is
    // still standard, so it never overwrites a figure somebody chose.
    if (waypoint === route[0] && wrote.altimeter_inhg != null
        && +$('altimeter').value === 29.92) {
      $('altimeter').value = wrote.altimeter_inhg;
    }
    fieldWx.set(waypoint.name, {
      station: report.station,
      valid_time: report.valid_time,
      sources: report.sources || {},
      notes: report.notes || [],
      wrote,
    });
    if (Object.keys(wrote).length) filled += 1;
  });

  renderWaypointList();

  // The sky between the fields: a window of forecast hours over every
  // waypoint, which the engine costs each leg against at both of its ends.
  const aloft = await getForecasts(arrivals);

  // Pass two: the fetched conditions change the density altitude, so every
  // distance and every chart reading has to be taken again -- and the winds
  // move the tops of climb, so this is the plan that has them in it.
  requestPlan();
  // Say which weather this was, not just how much of it. "Filled 2 fields"
  // reads the same whether it was observed now or forecast for tonight, and
  // those are very different things to be planning on.
  const when = arrivals.size ? ` for ${zulu(offBlocksUtc().toISOString())}+` : '';
  const legs = aloft ? `, ${aloft} point${aloft === 1 ? '' : 's'} aloft` : '';
  flashButton(button, failed.length
    ? `${filled} filled, ${failed.length} unavailable${legs}`
    : `Filled ${filled} field${filled === 1 ? '' : 's'}${legs}${when}`);
}

/** Fetch a window of forecast hours over every waypoint the pilot drew.
 *
 *  Over the waypoints rather than the leg midpoints, because each leg is
 *  costed at both of its ends and planned in whichever costs more. Two legs
 *  meeting at a waypoint share its forecast, so this is one request per
 *  point, not two per leg.
 *
 *  A window rather than a single hour. The engine settles the weather list
 *  and the navlog against each other -- the wind moves the times, the times
 *  move which forecast hour applies -- and it can only do that without a
 *  network call per pass if it already holds the hours. The window is the
 *  flight itself plus an hour at each end, so it is two or three hours for a
 *  local trip and never more than the server's cap.
 *
 *  Returns how many points came back. A point that fails is left out rather
 *  than faked: its legs fall back to the other end's forecast, and a leg with
 *  neither end to the wind typed on its row, and to calm under that.
 */
async function getForecasts(arrivals) {
  const drawn = route.filter((w) => !w.generated);
  if (drawn.length < 2) return 0;

  const start = offBlocksUtc();
  // How long the day is, from the plan we already have. An hour either side:
  // the fetch is anchored to the off-blocks hour, and the last leg is reached
  // after the total ETE, which the wind is about to change.
  const enRoute = lastPlan?.ok ? (lastPlan.totals?.time_min ?? 0) : 0;
  const hours = Math.min(MAX_FORECAST_HOURS, Math.ceil(enRoute / 60) + 2);

  let series;
  try {
    series = await api.aloftSeriesMany(
      drawn, start ? start.toISOString() : null, hours);
  } catch {
    clearForecasts();
    return 0;
  }

  // Every waypoint keeps its slot even when its fetch failed: the engine
  // reads the list positionally against the route, and a gap would shift
  // every forecast after it onto the wrong point.
  const notes = new Set();
  let got = 0;
  forecasts = drawn.map((w, n) => {
    const answer = series[n];
    const ok = answer && !answer.error && Array.isArray(answer.hours);
    if (!ok) return { name: w.name, lat: w.lat, lon: w.lon, hours: [] };
    got += 1;
    answer.hours.forEach((hour) => (hour.notes || []).forEach((x) => notes.add(x)));
    return {
      name: w.name,
      lat: w.lat,
      lon: w.lon,
      hours: answer.hours.map((hour) => ({
        valid_time: hour.valid_time, levels: hour.levels,
      })),
    };
  });

  forecastWx = got
    ? { points: got, of: drawn.length, hours, notes: [...notes] }
    : null;
  if (!got) forecasts = [];
  return got;
}

/** What the last NOTAM search found, or why it could not look. */
let notamReport = null;

/** Search for NOTAMs along the route and show the ones about this flight.
 *
 *  Its own button rather than part of "Get weather": it is a separate service
 *  with its own credentials and its own failure, and a briefing that half
 *  worked should say which half.
 *
 *  The search is done against the plan, not the route, because three of the
 *  four filters need what only a plan knows -- the altitude over each stretch
 *  of ground and the time the aeroplane is there.
 */
async function getNotams() {
  const button = $('get-notams');
  if (route.filter((w) => !w.generated).length < 2) {
    flashButton(button, 'Add a route first');
    return;
  }
  button.disabled = true;
  button.textContent = 'Searching…';
  try {
    notamReport = await api.notams(planBody());
  } catch {
    notamReport = { ok: false, error: 'the NOTAM search could not be reached' };
  }
  button.disabled = false;
  renderNotams();
  if (!notamReport.ok) {
    flashButton(button, notamReport.needs_credentials ? 'Not configured' : 'Search failed');
    return;
  }
  // Both numbers on the button too: "4 of 137" says the filter is working in
  // a way "4 NOTAMs" does not.
  flashButton(button, `${notamReport.relevant} of ${notamReport.returned}`);
}

/** The briefing, worst first.
 *
 *  Each entry says why it survived the filter -- how far off track, at what
 *  altitude, over what times. A pilot who can see the reason can judge
 *  whether the filter was right, which is the only way a filter that hides
 *  things earns any trust.
 */
function renderNotams() {
  const block = $('notam-block');
  const list = $('notam-list');
  list.innerHTML = '';
  if (!notamReport) { block.hidden = true; return; }
  block.hidden = false;

  if (!notamReport.ok) {
    $('notam-note').textContent = '';
    const div = document.createElement('div');
    div.className = notamReport.needs_credentials ? 'alert model' : 'alert';
    div.textContent = notamReport.error;
    list.appendChild(div);
    return;
  }

  // What was searched: the aerodromes and centres SkyLink was asked about.
  const idents = notamReport.designators?.length || 0;
  $('notam-note').textContent =
    `${notamReport.relevant} of ${notamReport.returned} within `
    + `${Math.round(notamReport.corridor_nm)} nm of track, from `
    + `${idents} identifier${idents === 1 ? '' : 's'} via SkyLink`;

  if (!notamReport.complete) {
    const div = document.createElement('div');
    div.className = 'alert';
    div.textContent = 'This briefing is incomplete — part of the search failed: '
      + notamReport.failed.join('; ')
      + '. Do not read the list below as "nothing else to report".';
    list.appendChild(div);
  }

  if (!notamReport.notams.length) {
    const div = document.createElement('div');
    div.className = 'hint notam-none';
    div.textContent = notamReport.returned
      ? `Nothing along this route: all ${notamReport.returned} found nearby were `
        + 'outside the corridor, at other altitudes, or not in force at the time.'
      : 'Nothing was returned for this route. Check against an official '
        + 'briefing before treating that as "no NOTAMs".';
    list.appendChild(div);
    return;
  }

  for (const one of notamReport.notams) {
    const item = document.createElement('div');
    item.className = `notam ${one.priority}`;
    const when = notamWhen(one);
    item.innerHTML =
      `<div class="notam-head">` +
        `<span class="pill ${one.priority}">${one.priority}</span>` +
        `<span class="notam-id">${one.location || '—'} ${one.number || ''}</span>` +
        `<span class="notam-where">${one.distance_nm == null ? 'unplaced'
          : one.distance_nm < 0.1 ? 'on track' : `${one.distance_nm} nm off track`}` +
          `</span>` +
      `</div>` +
      `<div class="notam-text">${escapeHtml(one.text)}</div>` +
      `<div class="notam-why">${when}${when && one.reasons.length ? ' · ' : ''}` +
        `${one.reasons.join(' · ')}</div>`;
    list.appendChild(item);
  }
}

/** A NOTAM's active window, in the terms it was published in.
 *
 *  The day is shown whenever the window crosses one. Bare Zulu times are what
 *  a pilot reads everywhere else on this page, but "0802Z to 0802Z" for a
 *  three-day closure reads as a window of no length at all -- which is the
 *  opposite of what it says.
 */
function notamWhen(one) {
  if (one.permanent) return 'permanent';
  if (!one.effective_start && !one.effective_end) return '';
  // Compared on the whole date, not the day of the month: 1 Sep and 1 Oct
  // share a day-of-month, and treating them as the same day printed a
  // month-long closure as "0000Z to 0000Z".
  const utcDate = (iso) => new Date(iso).toISOString().slice(0, 10);
  const utcMonth = (iso) => new Date(iso).toISOString().slice(0, 7);
  const both = one.effective_start && one.effective_end;
  const sameDay = both && utcDate(one.effective_start) === utcDate(one.effective_end);
  const sameMonth = both && utcMonth(one.effective_start) === utcMonth(one.effective_end);
  const stamp = (iso) => (sameDay ? zulu(iso) : sameMonth ? zuluDay(iso) : zuluDate(iso));
  const from = one.effective_start ? stamp(one.effective_start) : '—';
  const to = one.effective_end
    ? stamp(one.effective_end) + (one.estimated_end ? ' est' : '') : 'UFN';
  return `${from} to ${to}`;
}

/** A Zulu time with the day of the month on it: "04/0802Z". */
function zuluDay(iso) {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return '';
  return `${String(when.getUTCDate()).padStart(2, '0')}/${zulu(iso)}`;
}

/** With the month as well, for a window that crosses one: "01 Sep 0000Z".
 *  A bare day would make 1 Sep to 1 Oct read as "01/0000Z to 01/0000Z". */
function zuluDate(iso) {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return '';
  const month = when.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' });
  return `${String(when.getUTCDate()).padStart(2, '0')} ${month} ${zulu(iso)}`;
}

/** NOTAM text is somebody else's, and it goes into innerHTML. */
function escapeHtml(text) {
  const node = document.createElement('div');
  node.textContent = text || '';
  return node.innerHTML;
}

const flashTimers = new Map();

/** Say what happened on the button itself, then put its label back.
 *
 *  The result belongs where the click was, not in a banner across the panel:
 *  it is one line of outcome and it stops being interesting immediately. */
function flashButton(button, message) {
  button.textContent = message;
  clearTimeout(flashTimers.get(button));
  flashTimers.set(button, setTimeout(() => {
    button.textContent = button.dataset.label;
  }, 2600));
}

$('get-weather').dataset.label = 'Get weather';
$('get-weather').addEventListener('click', getWeather);

$('get-notams').dataset.label = 'Get NOTAMs';
$('get-notams').addEventListener('click', getNotams);

// --- the navlog as CSV, pulled down over the map ------------------------
//
// The same file "Download CSV" saves, shown before it is saved: the navlog's
// own columns, with the legs named by where they start and end. Fetched from
// the server rather than rebuilt here, so the drawer and the file can never
// disagree.

/** RFC 4180, which is what Python's csv module writes. */
function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i += 1; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i += 1;
      row.push(cell); rows.push(row); row = []; cell = '';
    } else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

let csvText = null;
let csvFilename = 'navlog.csv';
let csvRequest = 0;

function csvDrawerOpen() {
  return $('csv-handle').getAttribute('aria-expanded') === 'true';
}

async function refreshCsvDrawer() {
  if (!csvDrawerOpen()) return;
  const table = $('csv-table');
  const empty = $('csv-empty');
  const ticket = ++csvRequest;
  if (!lastPlan?.ok) {
    csvText = null;
    table.hidden = true;
    empty.hidden = false;
    empty.textContent = route.length < 2
      ? 'Add a departure and a destination to build a navlog.'
      : (lastPlan?.error || 'This route does not plan yet.');
    $('csv-note').textContent = '';
    return;
  }
  const { blob, filename, error } = await api.navlogCsv(planBody());
  // A newer plan asked while this one was on the wire; it will draw.
  if (ticket !== csvRequest) return;
  if (error) {
    csvText = null;
    table.hidden = true;
    empty.hidden = false;
    empty.textContent = error;
    return;
  }
  csvText = await blob.text();
  csvFilename = filename;
  const [head, ...rows] = parseCsv(csvText);
  // A column empty on every row -- ETA before an off-blocks time is set --
  // is left out of the view by default. The file keeps it.
  const shown = head.map((_, i) => $('csv-blank').checked || rows.some((r) => r[i]));
  const pick = (cells) => cells.filter((_, i) => shown[i]);
  const names = pick(head);
  table.querySelector('thead').innerHTML =
    `<tr>${names.map((h) => `<th>${escapeHtml(h)}</th>`).join('')}</tr>`;
  table.querySelector('tbody').innerHTML = rows.map((cells) => {
    const phase = cells[head.indexOf('Phase')] || '';
    const cls = cells[0] === 'Total' ? 'totals' : `phase-row-${phase}`;
    const text = new Set(['Leg Start', 'Leg End', 'Phase']);
    return `<tr class="${cls}">${pick(cells).map((c, i) =>
      `<td class="${text.has(names[i]) ? '' : 'num'}">${escapeHtml(c)}</td>`).join('')}</tr>`;
  }).join('');
  table.hidden = false;
  empty.hidden = true;
  const drawn = pilotLegs();
  $('csv-title').textContent = drawn.length
    ? `Navlog ${drawn[0].from.name} → ${drawn[drawn.length - 1].to.name}` : 'Navlog';
  $('csv-note').textContent = 'Time Rem: minutes left after each row, pattern included';
}

/** Open or close the drawer, dropping any height left over from a drag. */
function setCsvDrawer(open) {
  const drawer = $('csv-drawer');
  $('csv-handle').setAttribute('aria-expanded', String(open));
  $('csv-handle').textContent = open ? 'Navlog ▴' : 'Navlog ▾';
  drawer.style.height = '';
  $('map-wrap').style.removeProperty('--csv-drawer-h');
  drawer.classList.toggle('open', open);
  if (open) {
    drawer.hidden = false;
    refreshCsvDrawer();
  } else {
    // Hidden after the slide up, so it takes no clicks meant for the map.
    setTimeout(() => { if (!csvDrawerOpen()) drawer.hidden = true; }, 200);
  }
}

// Pulled down like a blind: the drawer follows the finger, and on release
// it opens if it was pulled a quarter of the way, closes otherwise. A plain
// click toggles it.
{
  const handle = $('csv-handle');
  let drag = null;
  handle.addEventListener('pointerdown', (event) => {
    drag = { y: event.clientY, moved: false, open: csvDrawerOpen() };
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener('pointermove', (event) => {
    if (!drag) return;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.abs(dy) < 6) return;
    const drawer = $('csv-drawer');
    const max = window.innerHeight / 2;
    if (!drag.moved) {
      drag.moved = true;
      drawer.hidden = false;
      drawer.classList.add('dragging');
      if (!drag.open) {
        // Filled while it is still being pulled, so there is a log to see.
        $('csv-handle').setAttribute('aria-expanded', 'true');
        refreshCsvDrawer();
      }
    }
    const start = drag.open ? drawer.getBoundingClientRect().height || max : 0;
    drag.start ??= start;
    const height = Math.max(0, Math.min(max, drag.start + dy));
    drawer.style.height = `${height}px`;
    $('map-wrap').style.setProperty('--csv-drawer-h', `${height}px`);
  });
  const release = () => {
    if (!drag) return;
    const drawer = $('csv-drawer');
    drawer.classList.remove('dragging');
    if (!drag.moved) setCsvDrawer(!drag.open);
    else setCsvDrawer(drawer.getBoundingClientRect().height > window.innerHeight / 8);
    drag = null;
  };
  handle.addEventListener('pointerup', release);
  handle.addEventListener('pointercancel', release);
  handle.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && csvDrawerOpen()) setCsvDrawer(false);
  });
}


$('csv-blank').addEventListener('change', () => refreshCsvDrawer());

$('csv-copy').addEventListener('click', async () => {
  if (!csvText) return;
  await navigator.clipboard.writeText(csvText);
  flashButton($('csv-copy'), 'Copied');
});
$('csv-copy').dataset.label = 'Copy CSV';

$('download-csv').addEventListener('click', () => {
  // The file is what the drawer shows, saved as it stands.
  if (!csvText) return;
  saveBlob(new Blob([csvText], { type: 'text/csv;charset=utf-8' }), csvFilename);
});

$('copy').addEventListener('click', async () => {
  if (!lastPlan?.ok) return;
  await navigator.clipboard.writeText(lastPlan.text);
  $('copy').textContent = 'Copied';
  setTimeout(() => { $('copy').textContent = 'Copy as text'; }, 1200);
});

// --- route file: the navlog panel's second tab -------------------------------

function setTab(name) {
  for (const button of document.querySelectorAll('.tabs [role="tab"]')) {
    const selected = button.dataset.tab === name;
    button.setAttribute('aria-selected', String(selected));
    $(button.getAttribute('aria-controls')).hidden = !selected;
  }
}
for (const button of document.querySelectorAll('.tabs [role="tab"]')) {
  button.addEventListener('click', () => setTab(button.dataset.tab));
}

/** Hand a file the server built to the browser, the way the CSV is. */
function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

$('download-kml').addEventListener('click', async () => {
  const button = $('download-kml');
  // A plan that does not solve today is still a mission worth keeping; the
  // server saves it without a profile.
  if (route.length < 2) {
    button.textContent = 'Nothing to export yet';
    setTimeout(() => { button.textContent = button.dataset.label; }, 1600);
    return;
  }
  const { blob, filename, error } = await api.routeKml(planBody());
  if (error) {
    button.textContent = 'Export failed';
    setTimeout(() => { button.textContent = button.dataset.label; }, 1600);
    return;
  }
  saveBlob(blob, filename);
});

/** What the last chosen file parsed to, waiting for "Load into route". Kept
 *  apart from the route so that choosing a file changes nothing until asked. */
let pendingImport = null;

function renderImportPreview(result) {
  const error = $('kml-error');
  const preview = $('kml-preview');
  const warnings = $('kml-warnings');
  preview.innerHTML = '';
  error.hidden = true;
  warnings.hidden = true;
  $('kml-load-row').hidden = true;
  pendingImport = null;
  if (!result) return;
  if (!result.ok) {
    error.textContent = result.error || 'could not read the file';
    error.hidden = false;
    return;
  }
  pendingImport = result;
  if (result.mission) {
    const m = result.mission;
    const pilot = m.waypoints.filter((w) => !w.generated);
    const events = pilot.reduce((n, w) => n + (w.events?.length || 0), 0);
    const li = document.createElement('li');
    li.className = 'mission';
    li.textContent = `Saved mission: ${m.planning_mode === 'manual' ? 'user-driven' : 'hybrid'}, `
      + `${m.cruise_altitude_ft} ft, ${events} leg event${events === 1 ? '' : 's'}, `
      + `${m.segment_overrides.length} leg edit${m.segment_overrides.length === 1 ? '' : 's'}`
      + (result.snapshot?.solved_at ? `, last solved ${zuluDate(result.snapshot.solved_at)}` : '')
      + '. Loading restores it and re-plans it in today\'s weather.';
    preview.appendChild(li);
  }
  for (const w of result.waypoints) {
    const li = document.createElement('li');
    const alt = w.altitude_ft != null ? `cross ${w.altitude_ft} ft` : '';
    const what = isAirport(w) ? (w.label || 'airport') : w.kind;
    li.innerHTML =
      `<span class="name">${escapeHtml(w.name)}</span>` +
      `<span class="meta">${w.lat.toFixed(4)}, ${w.lon.toFixed(4)}</span>` +
      `<span class="meta">${escapeHtml(what)}</span>` +
      (alt ? `<span class="meta">${alt}</span>` : '');
    preview.appendChild(li);
  }
  if (result.warnings?.length) {
    warnings.textContent = result.warnings.join(' ');
    warnings.hidden = false;
  }
  $('kml-load-row').hidden = false;
}

$('kml-file').addEventListener('change', async () => {
  const [file] = $('kml-file').files;
  if (!file) { renderImportPreview(null); return; }
  let result;
  try {
    result = await api.importRoute(file);
  } catch (err) {
    result = { ok: false, error: `could not read ${file.name}: ${err.message}` };
  }
  renderImportPreview(result);
});

$('kml-load').addEventListener('click', () => {
  if (!pendingImport?.waypoints?.length) return;
  const mode = document.querySelector('input[name="kml-mode"]:checked')?.value || 'replace';
  // A mission of ours replaces the plan outright: appending a whole plan's
  // settings to another route has no meaning.
  if (pendingImport.mission && mode === 'replace') {
    restoreMission(pendingImport.mission, pendingImport.snapshot);
    renderImportPreview(null);
    $('kml-file').value = '';
    setTab('navlog');
    return;
  }
  // Same defaults a point added from the search box gets; the file's own
  // segment type and crossing altitude win where it had them.
  let points = pendingImport.waypoints.map(({ label, ...w }) => ({
    segment_type: 'automatic', generated: false, id: newId(), ...w,
  }));
  route = mode === 'append' ? route : [];
  const last = route[route.length - 1];
  // A file that starts where the route ends is a continuation, not a
  // zero-length leg.
  if (last && points.length && Math.abs(points[0].lat - last.lat) < 1e-4
      && Math.abs(points[0].lon - last.lon) < 1e-4) {
    points = points.slice(1);
  }
  // Names stay unique across the whole route, see `isNameAvailable`. A file
  // drawn elsewhere can call two points the same thing; an airport visited
  // twice keeps its identifier, as it would if added twice from the search.
  for (const p of points) {
    if (!isAirport(p) && !isNameAvailable(p.name, p)) p.name = nextWaypointName();
    route.push(p);
  }
  renderImportPreview(null);
  $('kml-file').value = '';
  onRouteChanged();
  fitRoute();
  setTab('navlog');
});

// --- planning mode ------------------------------------------------------

function setPlanningMode(mode) {
  if (mode === 'auto') mode = 'hybrid';
  if (mode === planningMode) return;
  planningMode = mode;
  $('mode-hybrid').setAttribute('aria-pressed', String(mode === 'hybrid'));
  $('mode-manual').setAttribute('aria-pressed', String(mode === 'manual'));
  if (eventPick) setEventPick(null);

  // Taking over from an automatic pass means taking over its points too.
  // Resolution discards planner-owned points before re-deriving, so a TOC left
  // marked generated would simply vanish on the switch and leave the route
  // with nothing between departure and destination. Adopting them is the whole
  // reason the planner writes them into the list in the first place.
  if (mode === 'manual') {
    const drawn = pilotLegs();
    route = route.map((w) => (w.generated
      ? { ...w, generated: false, kind: 'waypoint', id: newId() }
      : w));
    // Each drawn leg is now several; what was typed on it holds on each part.
    for (const leg of drawn) {
      const first = route.indexOf(leg.from);
      const last = route.indexOf(leg.to);
      const parts = route.slice(first, last + 1);
      const keys = parts.slice(1).map((w, i) => `${parts[i].id}>${w.id}`);
      if (keys.length > 1) copyEdits(leg.key, keys);
    }
  }
  // In user-driven mode the profile comes from the declared segments, so the
  // cruise altitude is no longer what the aeroplane aims for.
  $('altitude-label').firstChild.textContent =
    mode === 'manual' ? 'Target cruise altitude ' : 'Cruise altitude ';
  $('altitude').disabled = mode === 'manual';
  // Switching modes can change which rows exist; as everywhere else, the
  // next plan is what decides which edits still have a leg to sit on.
  onRouteChanged();
}

$('mode-hybrid').addEventListener('click', () => setPlanningMode('hybrid'));

// --- saved missions -----------------------------------------------------
//
// A mission is what the pilot decided -- the fixes, each leg's events and
// edits, the flight's settings -- and never what was worked out from it. It
// travels in the exported KML (see `engine/mission.py`). Loading one puts the
// plan back exactly and plans it again; "Get weather" then re-solves it in
// the day's forecast, and the tops and bottoms of climb move to wherever the
// new wind puts them.

/** The solve the loaded mission was saved with, for `sinceSaved`. */
let savedSolve = null;

const MISSION_FIELDS = [
  ['cruise_altitude_ft', 'altitude'], ['cruise_rpm', 'rpm'], ['weight_lb', 'weight'],
  ['fuel_on_board_gal', 'fuel'], ['altimeter_inhg', 'altimeter'], ['isa_deviation_c', 'isadev'],
];

function restoreMission(mission, snapshot) {
  setEventPick(null);
  for (const [key, id] of MISSION_FIELDS) {
    if (mission[key] != null) $(id).value = mission[key];
  }
  $('night').checked = !!mission.night;
  if (mission.runway_margin != null) $('runway-margin').value = Math.round(mission.runway_margin * 100);
  if (mission.fuel_margin != null) $('fuel-margin').value = Math.round(mission.fuel_margin * 100);
  // The departure time is left as the pilot has it now: the saved one is
  // usually yesterday's guess, and the point of loading is to plan today.
  route = mission.waypoints
    .filter((w) => !w.generated)
    .map((w) => ({ ...w, events: (w.events || []).map((e) => ({ ...e })) }));
  segmentEdits = new Map();
  for (const { segment_key, phase, ...fields } of mission.segment_overrides || []) {
    const typed = Object.fromEntries(Object.entries(fields).filter(([, v]) => v != null));
    if (Object.keys(typed).length) segmentEdits.set(editKey(segment_key, phase), typed);
  }
  $('reset-edits').hidden = segmentEdits.size === 0;
  savedSolve = snapshot || null;
  // Set the mode without its side effects: the route is already the saved one.
  planningMode = mission.planning_mode === 'manual' ? 'manual' : 'hybrid';
  $('mode-hybrid').setAttribute('aria-pressed', String(planningMode === 'hybrid'));
  $('mode-manual').setAttribute('aria-pressed', String(planningMode === 'manual'));
  $('altitude-label').firstChild.textContent =
    planningMode === 'manual' ? 'Target cruise altitude ' : 'Cruise altitude ';
  $('altitude').disabled = planningMode === 'manual';
  // The field reports were not saved -- they are fetched again -- so the
  // provenance lines for the old ones go too.
  fieldWx = new Map();
  onRouteChanged();
  fitRoute();
  flashButton($('get-weather'), 'Get weather for today');
}

/** One line on what the day's plan changed against the saved one, or null.
 *
 *  Totals, and each top or bottom of climb and descent by how far it moved
 *  along the route -- matched by the leg it is on and its place among that
 *  leg's points of the same kind, since its name can change as others come
 *  and go. */
function sinceSaved(plan) {
  if (!savedSolve || !plan?.ok) return null;
  const t = plan.totals;
  const parts = [];
  const dt = t.time_min - savedSolve.total_time_min;
  const df = t.fuel_gal - savedSolve.total_fuel_gal;
  if (Math.abs(dt) >= 0.5) parts.push(`ETE ${dt > 0 ? '+' : ''}${dt.toFixed(0)} min`);
  if (Math.abs(df) >= 0.1) parts.push(`fuel ${df > 0 ? '+' : ''}${df.toFixed(1)} gal`);
  // Every top and bottom on the route, where it falls: a row's end role is
  // at its end, and a top of descent -- carried as the descent's start role --
  // at the end of the row before it.
  const points = (rows) => {
    const seen = new Map();
    const out = new Map();
    let before = 0;
    for (const row of rows) {
      for (const [role, name, at] of [
        [row.start_role, row.from, before], [row.end_role, row.to, row.cumulative_distance_nm]]) {
        if (!role || !row.segment_key) continue;
        const tag = `${row.segment_key}|${role}`;
        const n = (seen.get(tag) || 0) + 1;
        seen.set(tag, n);
        out.set(`${tag}|${n}`, { label: legLabel(name, role), at });
      }
      before = row.cumulative_distance_nm;
    }
    return out;
  };
  const was = points(savedSolve.rows);
  const now = points(plan.legs.filter((l) => l.covers_ground));
  for (const [key, point] of now) {
    const then = was.get(key);
    if (!then) continue;
    const moved = point.at - then.at;
    if (Math.abs(moved) >= 0.3) {
      parts.push(`${point.label} ${Math.abs(moved).toFixed(1)} nm ${moved > 0 ? 'later' : 'earlier'}`);
    }
  }
  const when = savedSolve.solved_at ? ` (saved ${zuluDate(savedSolve.solved_at)})` : '';
  return parts.length
    ? `Since the saved plan${when}: ${parts.join(', ')}.`
    : `Same as the saved plan${when}.`;
}
$('mode-manual').addEventListener('click', () => setPlanningMode('manual'));

// --- navlog consistency -------------------------------------------------

// One finding, one row. Shared by the sidebar list and the banner above the
// navlog so the two can never describe the same finding differently.
function findingRow(finding) {
  const div = document.createElement('div');
  div.className = `finding ${finding.severity}`;
  const where = finding.row == null ? 'plan' : `row ${finding.row + 1}`;
  div.innerHTML =
    `<span class="where">${where}</span><span>${finding.message}</span>`;
  return div;
}

// The banner over the navlog table. It carries findings only: a clean plan says
// so in the sidebar, and a permanent green bar would eat rows off a panel that
// is only 260px tall. Called from renderConsistency so clearConsistency, which
// renders null, empties it too -- findings must never outlive their plan.
function renderNavlogFindings(report) {
  const banner = $('navlog-findings');
  banner.innerHTML = '';
  const findings = (report && report.ok && report.findings) || [];
  banner.hidden = findings.length === 0;
  for (const finding of findings) banner.appendChild(findingRow(finding));
}

function renderConsistency(report) {
  renderNavlogFindings(report);
  const box = $('consistency');
  box.innerHTML = '';
  if (!report) {
    box.innerHTML =
      '<p class="hint">Not checked yet.</p>';
    return;
  }
  if (!report.ok) {
    const div = document.createElement('div');
    div.className = 'alert';
    div.textContent = report.error;
    box.appendChild(div);
    return;
  }
  if (!report.findings.length) {
    const div = document.createElement('div');
    div.className = 'finding info';
    div.innerHTML = '<span class="where">plan</span>' +
      '<span>Nothing inconsistent found.</span>';
    box.appendChild(div);
    return;
  }
  for (const finding of report.findings) box.appendChild(findingRow(finding));
}

$('check-consistency').addEventListener('click', async () => {
  if (route.length < 2) return;
  const button = $('check-consistency');
  button.disabled = true;
  button.textContent = 'Checking…';
  try {
    consistencyReport = await api.consistency(planBody());
    renderConsistency(consistencyReport);
  } finally {
    button.disabled = false;
    button.textContent = 'Check navlog consistency';
  }
});

renderConsistency(null);

// --- E6B bar ------------------------------------------------------------
//
// Two calculators that stand apart from the route: they answer a question
// about a place and a moment, not about the plan. Every number is computed by
// the same engine functions the navigation log uses -- nothing is worked out
// here -- so the two can never disagree.

/** A field's value as a number, or null if it is blank or not a number.
 *  Blank is a state, not a zero: an empty elevation box must not read as sea
 *  level and quietly produce an answer. */
function e6bValue(id) {
  const raw = $(id).value.trim();
  if (raw === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function e6bRow(key, value, className = '') {
  return `<div><span class="k">${key}</span>` +
         `<span class="v ${className}">${value}</span></div>`;
}

/** Show an engine refusal where the answer would have been. */
function e6bError(boxId, message) {
  const box = $(boxId);
  box.innerHTML = '';
  if (!message) return;
  const div = document.createElement('div');
  div.className = 'alert';
  div.textContent = message;
  box.appendChild(div);
}

const roundFt = (ft) => `${Math.round(ft).toLocaleString()} ft`;

/** Density altitude, from an elevation, an OAT and an altimeter setting. */
async function runDensityAltitude() {
  const elevationFt = e6bValue('da-elev');
  const oatC = e6bValue('da-oat');
  // The setting is the one field with a sensible default: 29.92 is what a
  // pilot who has not been given one would set anyway.
  const altimeterInhg = e6bValue('da-altimeter') ?? 29.92;
  e6bError('da-error', null);
  if (elevationFt === null || oatC === null) {
    $('da-out').innerHTML =
      '<p class="waiting">Enter an elevation and an OAT.</p>';
    return;
  }
  const result = await api.densityAltitude({ elevationFt, oatC, altimeterInhg });
  if (!result.ok) {
    $('da-out').innerHTML = '';
    e6bError('da-error', result.error);
    return;
  }
  $('da-out').innerHTML =
    e6bRow('Pressure alt', roundFt(result.pressure_altitude_ft), 'big') +
    e6bRow('Density alt', roundFt(result.density_altitude_ft), 'big') +
    e6bRow('ISA dev',
           `${result.isa_deviation_c >= 0 ? '+' : ''}` +
           `${result.isa_deviation_c.toFixed(1)} °C`) +
    e6bRow('Approx PA / DA',
           `${roundFt(result.pressure_altitude_approx_ft)} / ` +
           `${roundFt(result.density_altitude_approx_ft)}`, 'rule');
}

/** Magnetic variation at a latitude and longitude, positive east. */
async function runVariation() {
  const lat = e6bValue('var-lat');
  const lon = e6bValue('var-lon');
  e6bError('var-error', null);
  if (lat === null || lon === null) {
    $('var-out').innerHTML =
      '<p class="waiting">Enter a latitude and a longitude.</p>';
    return;
  }
  const result = await api.variation({ lat, lon });
  if (!result.ok) {
    $('var-out').innerHTML = '';
    e6bError('var-error', result.error);
    return;
  }
  // Spelled out as well as signed: "6.1° E" is what goes on the chart, and
  // the sign is what goes into the arithmetic.
  const variation = result.variation_deg;
  const hemisphere = variation >= 0 ? 'E' : 'W';
  $('var-out').innerHTML =
    e6bRow('Variation', `${Math.abs(variation).toFixed(1)}° ${hemisphere}`, 'big') +
    e6bRow('Signed', `${variation >= 0 ? '+' : ''}${variation.toFixed(2)}°`) +
    e6bRow('Dated', result.decimal_year.toFixed(2), 'rule');
}

/** Both calculators recompute as you type, so the last field entered finishes
 *  the answer. Debounced for the same reason the plan is: a held arrow key on
 *  a number input is a stream of keystrokes, not a stream of questions. */
function bindE6b(inputIds, run, clearId) {
  let timer = null;
  for (const id of inputIds) {
    $(id).addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(run, 180);
    });
  }
  $(clearId).addEventListener('click', () => {
    clearTimeout(timer);
    for (const id of inputIds) $(id).value = '';
    run();
  });
  run();
}

bindE6b(['da-elev', 'da-oat', 'da-altimeter'], runDensityAltitude, 'da-clear');
bindE6b(['var-lat', 'var-lon'], runVariation, 'var-clear');

// --- weight and balance -------------------------------------------------
//
// The one calculator here with a variable number of inputs: a loading form is
// as long as the aircraft has stations. The rows live in state rather than in
// the DOM so that adding or deleting one cannot lose what was typed in the
// others, and, as everywhere else, the sums come back from the engine.

/** The loading form. Name, weight in pounds, arm in inches aft of the datum;
 *  weight and arm are strings so a half-typed "-" or "" stays as the pilot
 *  left it. */
let wbRows = [];
let wbNextId = 0;

const WB_SEED_NAMES = ['Empty weight', 'Front seats', 'Fuel'];

function wbBlankRows() {
  return WB_SEED_NAMES.map((name) => ({
    id: wbNextId++, name, weight: '', arm: '',
  }));
}

/** A typed field as a number, or null if blank or not a number. Blank is a
 *  state, not a zero -- an empty weight box is a row not filled in yet, and
 *  must not be totalled as though it were. */
function wbNumber(raw) {
  const text = String(raw).trim();
  if (text === '') return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
}

/** The rows the engine can be asked about: both numbers present. */
function wbComplete() {
  return wbRows
    .map((row) => ({
      name: row.name,
      weight_lb: wbNumber(row.weight),
      arm_in: wbNumber(row.arm),
    }))
    .filter((row) => row.weight_lb !== null && row.arm_in !== null);
}

/** One input, built rather than templated: the station name is free text and
 *  has no business going anywhere near innerHTML. */
function wbInput(className, attrs, value, label) {
  const el = document.createElement('input');
  el.className = className;
  Object.assign(el, attrs);
  el.value = value;
  el.setAttribute('aria-label', label);
  return el;
}

function renderWeightBalance() {
  const box = $('wb-rows');
  box.innerHTML = '';
  for (const row of wbRows) {
    const line = document.createElement('div');
    line.className = 'wb-row';

    const nameEl = wbInput('wb-name', { type: 'text', placeholder: 'Station' },
                           row.name, 'Station name');
    const weightEl = wbInput('wb-weight',
                             { type: 'number', step: '1', inputMode: 'decimal',
                               placeholder: 'lb' },
                             row.weight, 'Weight, pounds');
    const armEl = wbInput('wb-arm',
                          { type: 'number', step: '0.1', inputMode: 'decimal',
                            placeholder: 'in' },
                          row.arm, 'Arm, inches');
    const del = document.createElement('button');
    del.className = 'wb-del';
    del.type = 'button';
    del.textContent = '×';
    del.title = 'Remove this row';
    del.setAttribute('aria-label', `Remove ${row.name || 'row'}`);

    nameEl.addEventListener('input', () => {
      row.name = nameEl.value;
      del.setAttribute('aria-label', `Remove ${row.name || 'row'}`);
    });
    weightEl.addEventListener('input', () => {
      row.weight = weightEl.value;
      wbRunSoon();
    });
    armEl.addEventListener('input', () => {
      row.arm = armEl.value;
      wbRunSoon();
    });
    // Deleting re-renders, which would throw away any half-typed name in
    // another row -- state holds every field, so nothing is lost.
    del.addEventListener('click', () => {
      wbRows = wbRows.filter((other) => other.id !== row.id);
      if (wbRows.length === 0) {
        wbRows = [{ id: wbNextId++, name: '', weight: '', arm: '' }];
      }
      renderWeightBalance();
      runWeightBalance();
    });

    line.append(nameEl, weightEl, armEl, del);
    box.appendChild(line);
  }
}

async function runWeightBalance() {
  const stations = wbComplete();
  e6bError('wb-error', null);
  if (stations.length === 0) {
    $('wb-out').innerHTML =
      '<p class="waiting">Enter a weight and an arm on at least one row.</p>';
    return;
  }
  const result = await api.weightBalance(stations);
  if (!result.ok) {
    $('wb-out').innerHTML = '';
    e6bError('wb-error', result.error);
    return;
  }
  $('wb-out').innerHTML =
    e6bRow('Gross weight',
           `${Math.round(result.gross_weight_lb).toLocaleString()} lb`, 'big') +
    e6bRow('CG', `${result.cg_in.toFixed(2)} in`, 'big') +
    e6bRow('Moment',
           `${Math.round(result.total_moment_in_lb).toLocaleString()} in-lb`,
           'rule');
}

let wbTimer = null;
/** Same debounce as the other two: a held arrow key is a stream of
 *  keystrokes, not a stream of questions. */
function wbRunSoon() {
  clearTimeout(wbTimer);
  wbTimer = setTimeout(runWeightBalance, 180);
}

$('wb-add').addEventListener('click', () => {
  wbRows.push({ id: wbNextId++, name: '', weight: '', arm: '' });
  renderWeightBalance();
  // Straight to the new row's name box: adding a row is always followed by
  // typing in it.
  const boxes = $('wb-rows').querySelectorAll('.wb-name');
  boxes[boxes.length - 1].focus();
});

$('wb-clear').addEventListener('click', () => {
  clearTimeout(wbTimer);
  wbRows = wbBlankRows();
  renderWeightBalance();
  runWeightBalance();
});

wbRows = wbBlankRows();
renderWeightBalance();
runWeightBalance();

// --- E6B bar visibility -------------------------------------------------
//
// A wide navigation log and a narrow screen are a common pair, and the two
// calculators are not needed while reading it. Hiding the bar gives the
// column back to the map and the log rather than leaving it blank.

function setE6bHidden(hidden) {
  const toggle = $('e6b-toggle');
  document.getElementById('app').classList.toggle('e6b-hidden', hidden);
  toggle.textContent = hidden ? '\u2039' : '\u203a';
  toggle.title = hidden ? 'Show the E6B bar' : 'Hide the E6B bar';
  toggle.setAttribute('aria-expanded', String(!hidden));
  // The map fills its container absolutely, so it has to be told the
  // container changed width or the canvas stays the old size.
  map.resize();
}

$('e6b-toggle').addEventListener('click', () => {
  setE6bHidden(!document.getElementById('app').classList.contains('e6b-hidden'));
});

$('wind-scope').addEventListener('change', () => { windScope = $('wind-scope').value; });

// --- the navigation log's height ----------------------------------------
//
// The log and the map share the middle column, and which needs the room
// changes through planning: the map while the route is drawn, the log once
// it is being read. So the split is the pilot's to drag. The height is kept
// between visits; the map is told whenever its container changes size, or its
// canvas stays the old one.

const NAVLOG_HEIGHT_KEY = 'e6b.navlogHeight';
const NAVLOG_DEFAULT_PX = 260;
const NAVLOG_MIN_PX = 120;
// The map is never squeezed below this: enough to see where the route goes.
const MAP_MIN_PX = 90;

function navlogMaxPx() {
  return Math.max(NAVLOG_MIN_PX, $('app').getBoundingClientRect().height - MAP_MIN_PX);
}

function navlogHeightPx() {
  return $('navlog-wrap').getBoundingClientRect().height;
}

let mapResizeFrame = 0;
function setNavlogHeight(px, { save = true } = {}) {
  const height = Math.round(Math.min(navlogMaxPx(), Math.max(NAVLOG_MIN_PX, px)));
  $('app').style.setProperty('--navlog-h', `${height}px`);
  const expanded = height >= navlogMaxPx() - 4;
  const button = $('navlog-size');
  button.textContent = expanded ? '▼ Restore' : '▲ Expand';
  button.title = expanded
    ? 'Give the map its space back'
    : 'Give the navigation log most of the screen';
  $('navlog-resize').setAttribute('aria-valuenow', String(height));
  if (save) {
    try { localStorage.setItem(NAVLOG_HEIGHT_KEY, String(height)); } catch { /* private mode */ }
  }
  cancelAnimationFrame(mapResizeFrame);
  mapResizeFrame = requestAnimationFrame(() => map.resize());
}

/** Tallest if it is not already, the default if it is. */
function toggleNavlogExpanded() {
  const expanded = navlogHeightPx() >= navlogMaxPx() - 4;
  setNavlogHeight(expanded ? NAVLOG_DEFAULT_PX : navlogMaxPx());
}

{
  const grip = $('navlog-resize');
  let drag = null;
  grip.addEventListener('pointerdown', (event) => {
    drag = { y: event.clientY, start: navlogHeightPx() };
    grip.setPointerCapture(event.pointerId);
    document.body.classList.add('resizing-navlog');
    event.preventDefault();
  });
  grip.addEventListener('pointermove', (event) => {
    if (!drag) return;
    // Up is taller: the log grows into the map.
    setNavlogHeight(drag.start + (drag.y - event.clientY), { save: false });
  });
  const release = () => {
    if (!drag) return;
    drag = null;
    document.body.classList.remove('resizing-navlog');
    setNavlogHeight(navlogHeightPx());
  };
  grip.addEventListener('pointerup', release);
  grip.addEventListener('pointercancel', release);
  grip.addEventListener('dblclick', toggleNavlogExpanded);
  grip.addEventListener('keydown', (event) => {
    const step = event.shiftKey ? 120 : 40;
    if (event.key === 'ArrowUp') setNavlogHeight(navlogHeightPx() + step);
    else if (event.key === 'ArrowDown') setNavlogHeight(navlogHeightPx() - step);
    else if (event.key === 'Home') setNavlogHeight(navlogMaxPx());
    else if (event.key === 'End') setNavlogHeight(NAVLOG_MIN_PX);
    else return;
    event.preventDefault();
  });
  $('navlog-size').addEventListener('click', toggleNavlogExpanded);

  let saved = NaN;
  try { saved = Number(localStorage.getItem(NAVLOG_HEIGHT_KEY)); } catch { /* private mode */ }
  if (Number.isFinite(saved) && saved > 0) setNavlogHeight(saved, { save: false });

  // A smaller window must not leave the map with nothing: re-clamp.
  window.addEventListener('resize', () => {
    const style = $('app').style.getPropertyValue('--navlog-h');
    if (style) setNavlogHeight(parseFloat(style), { save: false });
  });
}
