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
 *  type; "auto" lets the planner place TOC/TOD instead. Kept in step with the
 *  toggle's pressed state and the altitude field's disabled state in the
 *  markup, which start on the same mode. */
let planningMode = 'manual';
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
    const f = event.features[0];
    showAirportPopup(f.geometry.coordinates, f.properties);
  });
  map.on('click', 'vfr-dot', (event) => {
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
  route.push({ segment_type: 'automatic', generated: false, ...waypoint });
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
  route.splice(bestLeg + 1, 0, {
    name: `WP${route.length - 1}`,
    lat: +lngLat.lat.toFixed(5),
    lon: +lngLat.lng.toFixed(5),
    kind: 'waypoint',
    elevation_ft: null,
  });
  onRouteChanged();
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
  // Row edits are keyed by row index and a route change can renumber the rows,
  // but which ones it renumbered is only knowable once the next plan comes
  // back -- `pruneOverrides` does it there, per row, against the leg each edit
  // was typed on.
  //
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
        (planningMode === 'auto'
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
          `${waypoint.lat.toFixed(4)}, ${waypoint.lon.toFixed(4)}</div>`;
    li.innerHTML =
      `<div class="wp-row">` +
        `<span class="drag" title="Drag to reorder">⠿</span>` +
        `<span class="seq">${index + 1}</span>` +
        `<span class="name">${waypoint.name}</span>` +
        segment +
        landing +
        `<button class="remove" type="button" title="Remove">×</button>` +
      `</div>` + coords + fieldBlock(waypoint, index);
    li.querySelector('.remove').addEventListener('click', () => removeWaypoint(index));

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

// --- manual row edits ---------------------------------------------------
//
// Keyed by row index, which is what the engine's `overrides` map wants -- but
// a row index is only a position, and positions move. Each entry therefore
// also remembers the leg it was typed on (the two points and the phase), and
// every plan checks that the row at that index is still that leg. An edit
// survives anything that leaves its leg alone -- a field temperature, an
// altimeter setting, a waypoint dragged somewhere else -- and is dropped, with
// a notice, only when the leg it belonged to is gone. Wind is entered here and
// nowhere else, so silently reapplying one to a different leg would be worse
// than losing it, and silently losing it is what made the table unusable.
//
// Each value is `{ leg, fields }`: `leg` the signature below, `fields` the
// numbers the pilot typed.

let overrides = new Map();

/** What makes a navlog row itself: where it goes, and what it is doing. */
function legSignature(leg) {
  return leg ? `${leg.from}>${leg.to}|${leg.phase}` : null;
}

/** Rows dropped by the last prune, for the notice under the table. */
let droppedRows = [];

/** Drop every edit whose leg is no longer at the index it was typed at.
 *
 *  Returns true if anything went, in which case the plan on screen was built
 *  with an edit that has since been withdrawn and has to be built again.
 */
function pruneOverrides(plan) {
  if (!overrides.size) return false;
  const legs = plan?.ok ? plan.legs : null;
  if (!legs) return false;
  const gone = [...overrides.entries()].filter(
    ([row, entry]) => entry.leg !== null && legSignature(legs[row]) !== entry.leg,
  );
  if (!gone.length) return false;
  gone.forEach(([row]) => overrides.delete(row));
  droppedRows = gone.map(([row]) => row + 1);
  $('reset-edits').hidden = overrides.size === 0;
  return true;
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
  return !role || name === role ? name : `${name} (${role})`;
}

function restoreFocus() {
  if (!focusedCell) return;
  const { row, field, start, end } = focusedCell;
  const tr = document.querySelectorAll('#navlog tbody tr')[row];
  const input = tr && tr.querySelector(`td.${FIELD_CLASS[field]} input`);
  if (!input) { focusedCell = null; return; }
  input.focus();
  try { input.setSelectionRange(start, end); } catch { /* number inputs vary */ }
}

function clearOverrides() {
  droppedRows = [];
  if (overrides.size === 0) return;
  overrides = new Map();
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

function setOverride(row, field, value) {
  const entry = overrides.get(row) || {
    // The leg this number is about, taken from the plan the pilot is looking
    // at as they type it.
    leg: legSignature(lastPlan?.ok ? lastPlan.legs[row] : null),
    fields: {},
  };
  if (value === null) {
    delete entry.fields[field];
  } else {
    entry.fields[field] = value;
  }
  if (Object.keys(entry.fields).length === 0) {
    overrides.delete(row);
  } else {
    overrides.set(row, entry);
  }
  $('reset-edits').hidden = overrides.size === 0;
  requestPlan();
}

function overridesPayload() {
  return [...overrides.entries()].map(([row, entry]) => ({ row, ...entry.fields }));
}

/** Turn a table cell into a number the pilot can type over. */
function makeEditable(cell, { row, field, value, format, title }) {
  cell.classList.add('editable');
  cell.title = title;
  const edited = overrides.get(row)?.fields[field] != null;
  if (edited) cell.classList.add('edited');

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
    setOverride(row, field, next);
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
    if (event.key === 'Escape') { focusedCell = null; setOverride(row, field, null); }
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
    overrides: overridesPayload(),
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
    // No route-wide wind: it is typed on the navlog row it applies to, and
    // travels to the engine in `overrides`.
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
  // The notice below the table belongs to the plan that dropped the edits and
  // to no later one, so it starts every plan empty.
  droppedRows = [];
  lastPlan = await api.plan(planBody());
  // An edit whose leg is gone was still sent, so this plan was built with a
  // number that no longer applies to the row it landed on. Drop it and plan
  // again rather than show it: one extra round trip against a wind on the
  // wrong leg is not a trade.
  if (pruneOverrides(lastPlan)) {
    lastPlan = await api.plan(planBody());
  }
  // Before adopting: adoption may re-render the sidebar, and the derived
  // lines should already be right when it does.
  adoptFieldAir(lastPlan?.ok ? lastPlan.resolved_waypoints : null);
  if (lastPlan?.ok) adoptResolvedRoute(lastPlan.resolved_waypoints);
  renderFieldAir();
  renderNavlog(lastPlan);
  renderRouteLine();
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
    // Keep anything of the pilot's the engine does not round-trip.
    ...(w.generated ? {} : route.find((r) => r.name === w.name) || {}),
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
  plan.legs.forEach((leg, row) => {
    const tr = document.createElement('tr');
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
        // Course through ground speed, plus distance: ten columns of nothing.
        '<td class="num">—</td>'.repeat(10) +
        `<td class="num">${round(leg.ete_min, 1)}</td>` +
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
      `<td class="num tas"></td>` +
      `<td class="num">${round(leg.ground_speed_kt)}</td>` +
      `<td class="num">${round(leg.distance_nm, 1)}</td>` +
      `<td class="num">${round(leg.ete_min, 1)}</td>` +
      `<td class="num">${round(leg.fuel_gal, 1)}</td>` +
      `<td class="num">${round(leg.fuel_remaining_gal, 1)}</td>`;

    // The cells the pilot can overwrite. Everything else on the row is
    // derived from them and recomputes on the next plan.
    makeEditable(tr.querySelector('.wind-dir'), {
      row, field: 'wind_from_deg', value: leg.wind_from_deg,
      format: (v) => Math.round(v),
      title: 'Wind direction on this leg, degrees true. Blank for calm.',
    });
    makeEditable(tr.querySelector('.wind-speed'), {
      row, field: 'wind_speed_kt', value: leg.wind_speed_kt,
      format: (v) => Math.round(v),
      title: 'Wind speed on this leg, knots. Blank for calm.',
    });
    // The altitude this leg *ends* at, which carries forward to every row
    // after it -- editing the top of a climb re-flies the rest of the plan.
    makeEditable(tr.querySelector('.alt'), {
      row, field: 'altitude_ft',
      value: leg.exit_altitude_ft ?? leg.altitude_ft,
      format: (v) => Math.round(v),
      title: 'Altitude at the end of this leg. Blank to compute it again.',
    });
    // Temperature is the odd one out: it describes the air, not the row, so
    // it joins the field temperatures in the flight's profile and lapses into
    // the legs above and below rather than stopping at this one.
    makeEditable(tr.querySelector('.oat'), {
      row, field: 'oat_c', value: leg.oat_c,
      format: (v) => Math.round(v),
      title: 'Outside air temperature at this altitude, °C — the FD forecast '
        + 'figure for this part of the route. It lapses into the rows above '
        + 'and below. Blank to go back to the standard lapse.',
    });
    // Pressure altitude, which the route's altimeter setting normally decides.
    // Typing one says the air over this leg does not match that setting; the
    // density altitude beside it and the charts this row reads follow.
    makeEditable(tr.querySelector('.pa'), {
      row, field: 'pressure_altitude_ft', value: leg.pressure_altitude_ft,
      format: (v) => Math.round(v),
      title: 'Pressure altitude of this row\'s air, ft — what the altimeter '
        + 'reads with 29.92 set. Density altitude and this row\'s chart '
        + 'readings follow from it. Blank to take it from the altimeter '
        + 'setting again.',
    });
    makeEditable(tr.querySelector('.tas'), {
      row, field: 'tas_kt', value: leg.tas_kt,
      format: (v) => Math.round(v),
      title: 'True airspeed, knots. Blank to go back to the POH figure.',
    });
    body.appendChild(tr);
  });

  const t = plan.totals;
  const tr = document.createElement('tr');
  tr.innerHTML =
    `<td colspan="18">Total</td>` +
    `<td class="num">${round(t.distance_nm, 1)}</td>` +
    `<td class="num">${round(t.time_min, 1)}</td>` +
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
  // Edits this browser dropped, said in the same place as the engine's own
  // warnings: a number the pilot typed disappearing without a word is what
  // makes a navlog untrustworthy.
  if (droppedRows.length) {
    const div = document.createElement('div');
    div.className = 'alert';
    div.textContent = `manual edits on row(s) ${droppedRows.join(', ')} were `
      + 'dropped: the route no longer has those legs';
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

$('download-csv').addEventListener('click', async () => {
  if (!lastPlan?.ok) return;
  const button = $('download-csv');
  const { blob, filename, error } = await api.navlogCsv(planBody());
  if (error) {
    button.textContent = 'Export failed';
    setTimeout(() => { button.textContent = 'Download CSV'; }, 1600);
    return;
  }
  // Handed to the browser as a blob URL and revoked straight after: the file
  // is built per click from the plan on screen, so keeping the URL alive would
  // only pin a stale copy in memory.
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
});

$('copy').addEventListener('click', async () => {
  if (!lastPlan?.ok) return;
  await navigator.clipboard.writeText(lastPlan.text);
  $('copy').textContent = 'Copied';
  setTimeout(() => { $('copy').textContent = 'Copy as text'; }, 1200);
});

// --- planning mode ------------------------------------------------------

function setPlanningMode(mode) {
  if (mode === planningMode) return;
  planningMode = mode;
  $('mode-auto').setAttribute('aria-pressed', String(mode === 'auto'));
  $('mode-manual').setAttribute('aria-pressed', String(mode === 'manual'));

  // Taking over from an automatic pass means taking over its points too.
  // Resolution discards planner-owned points before re-deriving, so a TOC left
  // marked generated would simply vanish on the switch and leave the route
  // with nothing between departure and destination. Adopting them is the whole
  // reason the planner writes them into the list in the first place.
  if (mode === 'manual') {
    route = route.map((w) => (w.generated
      ? { ...w, generated: false, kind: 'waypoint' }
      : w));
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

$('mode-auto').addEventListener('click', () => setPlanningMode('auto'));
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
