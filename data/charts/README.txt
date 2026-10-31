FAA VFR raster charts -- downloaded by hand, not committed.

Get them from https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/vfr/
and unzip each download into the folder for its series:

    data/charts/sectional/<anything>/<Name> SEC.tif   + .tfw + .htm
    data/charts/tac/<anything>/<Name> TAC.tif         + .tfw + .htm
                                <Name> FLY.tif        (the flyway planning side)

The first folder level is the series the map's layer menu groups by. Below
that, any layout works: every .tif under it is found (engine/charts.py).

Keep the .htm beside each .tif. It is the FAA's metadata for that edition
and is where the chart's effective and expiry dates come from; without it
the chart still draws, but the currency list cannot say when it lapses.
VFR charts are reissued every 56 days.

Tiles rendered from the charts land in two places, neither of them
committed:

  data/charts/tiles/     what `make charts` built, from the sources here
  .cache/chart-tiles/    what the dev server rendered as you panned past
                         the committed zoom cap

Both are ignored. The GeoTIFFs above are the data; tiles are derived from
them in about a minute and would otherwise store the same charts twice.
Only `make charts` writes the first directory, so the two never mix.

    make charts                 # native zoom
    make charts MAXZOOM=11      # smaller and faster, soft past z11

MapLibre overzooms past the top level rather than showing nothing, so a
capped chart goes soft rather than blank.

Because the tiles are not committed, anything serving this app from a fresh
checkout has to build them. See docs/ARCHITECTURE.md section 4e.
