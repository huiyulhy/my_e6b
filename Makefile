# Every recipe clears PYTHONPATH first.
#
# Sourcing /opt/ros/humble/setup.bash puts ROS 2's Python 3.10 site-packages on
# PYTHONPATH for every shell. Those leak into this project's 3.13 venv, and
# pytest then autoloads ROS's launch_testing plugin, which fails on import. The
# venv itself is fine -- only the inherited PYTHONPATH is the problem.

UV := PYTHONPATH= uv

.PHONY: test lint fmt validate serve check data airports basemap airspace charts ui-shot requirements

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .

fmt:
	$(UV) run ruff format .

check: lint test

# Re-export the runtime-only pin set from uv.lock, for hosts that install with
# pip rather than uv. Run after changing dependencies and commit the result.
requirements:
	$(UV) export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt

# Dump every published POH cell for manual diff against a real POH.
validate:
	$(UV) run python tools/validate_poh.py

serve:
	$(UV) run uvicorn server.main:app --reload --port 8137

# --- data pipeline ---------------------------------------------------------
# Source files are cached in .cache/, so re-running is cheap. Pass --refresh
# to re-download: `$(UV) run python tools/build_airports.py --refresh`.

data: airports basemap airspace

airports:
	$(UV) run python tools/build_airports.py

basemap:
	$(UV) run python tools/build_basemap.py

# Reads the shapefile already inside the NASR distribution, so unlike the
# targets above this one downloads nothing.
airspace:
	$(UV) run python tools/build_airspace.py

# Pre-render every chart under data/charts/ into Web Mercator PNG tiles, plus
# the manifest that describes them. This is what puts charts on the deployed
# service: it has no GeoTIFFs, so committed tiles are the only form a chart
# reaches it in. MAXZOOM caps the pyramid -- the top level is most of the
# bytes, and MapLibre overzooms past whatever the manifest advertises.
#
#     make charts              # native zoom
#     make charts MAXZOOM=11   # smaller, softer past z11
charts:
	$(UV) run python tools/build_charts.py $(if $(MAXZOOM),--max-zoom $(MAXZOOM),)

# Drive the map in a real browser and screenshot it. MapLibre needs WebGL, so
# this is the only way to catch a broken layer style -- it fails at runtime,
# not at import. Needs `make serve` running.
ui-shot:
	$(UV) run python tools/screenshot_ui.py

# How far the altimeter setting moves over a day, and what that is worth in
# feet of pressure altitude. This is the evidence behind fetching a live
# setting rather than planning on 29.92. Downloads METARs; cached in .cache/.
altimeter-trend:
	$(UV) run python tools/plot_altimeter_trend.py
