"""Build the Class B/C/D airspace overlay from the NASR shapefile.

    make airspace

The boundaries ship with NASR already -- `Additional_Data/Shape_Files/` -- so
nothing is downloaded here. The source is 371 MB of very densely sampled
polygons: 12.1 million vertices for 5,614 records, some of them 6,000 points
for a shape a few miles across. Simplifying is what makes this a map layer
instead of a download.

Only B, C and D are kept. Those are the airspace that changes what a VFR pilot
does -- a clearance for B, two-way communication for C and D. Class E is 4,327
of the records and 8 of the 12 million vertices, and 3,328 of those are the
CLASS_E5 700 ft AGL transition blanket that covers most of the country; drawn
as outlines it is a wash rather than information, which is why sectionals
render it as a soft vignette instead.

TOLERANCE IS A SAFETY-ADJACENT CHOICE HERE, unlike the coastline in
build_basemap.py. 0.001 degrees is about 111 m, roughly 1.4% of the radius of a
typical 4.4 nm Class D. That is fine for showing where airspace is and wrong
for deciding whether you are inside it -- the app says "not for navigation" and
means it. Do not loosen this without understanding that trade.

NAD83 and WGS84 differ by well under a metre in the US, so the coordinates are
used as-is with no reprojection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import shapefile
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "nasr" / "Additional_Data" / "Shape_Files" / "Class_Airspace"
OUT = ROOT / "data" / "aero" / "airspace.geojson"

KEEP_CLASSES = ("B", "C", "D")

# About 111 m. See the module docstring before changing this.
TOLERANCE = 0.001

# Four decimals is about 11 m, finer than the simplification above, so rounding
# costs nothing that the tolerance has not already given up.
DIGITS = 4


def build() -> int:
    if not SOURCE.with_suffix(".shp").exists():
        print(f"missing {SOURCE.with_suffix('.shp')}", file=sys.stderr)
        print("The NASR distribution provides it; see docs/ARCHITECTURE.md.", file=sys.stderr)
        return 1

    reader = shapefile.Reader(str(SOURCE))
    fields = [f[0] for f in reader.fields[1:]]
    source_size = SOURCE.with_suffix(".shp").stat().st_size

    features = []
    counts = dict.fromkeys(KEEP_CLASSES, 0)
    vertices_before = vertices_after = 0

    # Streamed rather than read whole: the shapefile is 371 MB.
    for record in reader.iterShapeRecords():
        row = dict(zip(fields, record.record))
        airspace_class = row["CLASS"]
        if airspace_class not in KEEP_CLASSES:
            continue

        geometry = shape(record.shape.__geo_interface__)
        # NASR rings are not consistently wound, which shapely reads as
        # self-intersecting. buffer(0) repairs that without moving the edge.
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        simplified = geometry.simplify(TOLERANCE, preserve_topology=True)
        if simplified.is_empty:
            continue

        vertices_before += len(record.shape.points)
        vertices_after += _count_coordinates(mapping(simplified)["coordinates"])
        counts[airspace_class] += 1

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": row["NAME"],
                    "class": airspace_class,
                    # Kept as published: a floor is either "SFC" or a number of
                    # feet MSL, and collapsing the two loses the distinction.
                    "lower": row["LOWER_VAL"],
                    "lower_code": row["LOWER_CODE"],
                    "upper": row["UPPER_VAL"],
                },
                "geometry": mapping(simplified),
            }
        )

    for feature in features:
        feature["geometry"]["coordinates"] = _round(feature["geometry"]["coordinates"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"))
    )

    size = OUT.stat().st_size
    print(f"  {' '.join(f'{k}={v}' for k, v in counts.items())}")
    print(f"  {vertices_before} vertices -> {vertices_after} ({TOLERANCE} deg tolerance)")
    print(f"  {len(features)} features -> {OUT.relative_to(ROOT)} ({size // 1024} KB)")
    print(f"  from {source_size // 1024 // 1024} MB of shapefile.")
    print("FAA NASR, public domain. Not for navigation.")
    return 0


def _count_coordinates(node) -> int:
    if isinstance(node, (list, tuple)):
        if node and isinstance(node[0], (int, float)):
            return 1
        return sum(_count_coordinates(item) for item in node)
    return 0


def _round(node):
    if isinstance(node, (list, tuple)):
        if node and isinstance(node[0], (int, float)):
            return [round(value, DIGITS) for value in node]
        return [_round(item) for item in node]
    return node


if __name__ == "__main__":
    raise SystemExit(build())
