"""Build the offline basemap from Natural Earth.

    make basemap

MapLibre renders GeoJSON directly, so for a continental-scale VFR planner a
few simplified vector layers are enough and no tile pipeline is needed. The
result is a couple of megabytes that ship in the repo and work in airplane
mode -- the whole point of the project.

Natural Earth is public domain. Source data is clipped to the US bounding box
and simplified; state boundaries, coastline, large lakes and the interstate-scale
highway network give enough geographic context to recognise where a route goes.
Sectional raster charts are a separate, much larger, optional layer (see
docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from shapely.geometry import box, mapping, shape

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "basemap"
CACHE = ROOT / ".cache"

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"

# Continental US plus enough margin for Alaska-adjacent and coastal work.
CLIP = box(-172.0, 18.0, -66.0, 72.0)

# Keeping every Natural Earth attribute would triple the file size for
# properties the map never reads.
KEEP_PROPERTIES = ("name", "name_en", "postal", "admin")

LAYERS = {
    # name: (source file, simplification tolerance in degrees, properties to
    # keep, optional predicate on the source properties)
    "states": ("ne_50m_admin_1_states_provinces_lakes", 0.02, KEEP_PROPERTIES, None),
    "coastline": ("ne_50m_coastline", 0.01, KEEP_PROPERTIES, None),
    "lakes": ("ne_50m_lakes", 0.01, KEEP_PROPERTIES, None),
    # Highways are a primary VFR pilotage reference. The source is the whole
    # world at 56,600 features; restricting it to US major highways leaves
    # about 2,000, which is the difference between half a megabyte and fifty.
    # Secondary highways would triple that for detail below VFR planning zoom.
    # `name` is the route number and `level` distinguishes Interstate from US
    # and state routes, which is what the map needs to name a road under the
    # cursor. `type` is deliberately not kept: the filter already pins it to
    # "Major Highway", so storing it would repeat one constant 2,000 times.
    "highways": (
        "ne_10m_roads",
        0.01,
        ("name", "level"),
        lambda p: p.get("sov_a3") == "USA" and p.get("type") == "Major Highway",
    ),
}


def fetch(name: str, refresh: bool = False) -> Path:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.geojson"
    if path.exists() and not refresh:
        print(f"  using cached {path.name} ({path.stat().st_size // 1024} KB)")
        return path
    url = f"{BASE}/{name}.geojson"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())
    print(f"  saved {path.name} ({path.stat().st_size // 1024} KB)")
    return path


def build(refresh: bool = False) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_before = total_after = 0

    for layer, (source, tolerance, keep, where) in LAYERS.items():
        print(f"\n{layer}:")
        path = fetch(source, refresh)
        raw = json.loads(path.read_text())
        total_before += path.stat().st_size

        features = []
        for feature in raw["features"]:
            # A dict test before any shapely work: for the road layer this
            # discards seven eighths of the source before it is ever parsed.
            if where is not None and not where(feature["properties"]):
                continue
            geometry = shape(feature["geometry"])
            if not geometry.intersects(CLIP):
                continue
            clipped = geometry.intersection(CLIP)
            if clipped.is_empty:
                continue
            # Simplification is why this fits in the repo. The tolerances are
            # far below what matters at VFR planning zoom levels.
            simplified = clipped.simplify(tolerance, preserve_topology=True)
            if simplified.is_empty:
                continue
            properties = {
                key: feature["properties"][key]
                for key in keep
                if feature["properties"].get(key)
            }
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": mapping(simplified),
                }
            )

        out = OUT_DIR / f"{layer}.geojson"
        # Coordinates are rounded to five decimals, about a metre, which is
        # far finer than anything drawn at these zoom levels and materially
        # smaller than full float precision.
        out.write_text(
            json.dumps(
                {"type": "FeatureCollection", "features": features},
                separators=(",", ":"),
            )
        )
        _round_coordinates(out, 5)
        size = out.stat().st_size
        total_after += size
        print(f"  {len(features)} features -> {out.relative_to(ROOT)} ({size // 1024} KB)")

    print(
        f"\nBasemap total {total_after // 1024} KB, from {total_before // 1024} KB of source."
    )
    print("Natural Earth, public domain.")
    return 0


def _round_coordinates(path: Path, digits: int) -> None:
    """Re-emit the file with coordinates rounded, to shrink it."""
    data = json.loads(path.read_text())

    def walk(node):
        if isinstance(node, list):
            if node and isinstance(node[0], (int, float)):
                return [round(value, digits) for value in node]
            return [walk(item) for item in node]
        return node

    for feature in data["features"]:
        feature["geometry"]["coordinates"] = walk(feature["geometry"]["coordinates"])
    path.write_text(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(build(refresh="--refresh" in sys.argv))
