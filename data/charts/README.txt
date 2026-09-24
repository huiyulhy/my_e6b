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

Tiles rendered from the charts land in data/charts/tiles/. The dev server
fills that on demand as you pan; `make charts` fills all of it and writes
manifest.json beside it.

Unlike the charts, the tiles ARE committed. A deployed instance has no
GeoTIFFs on it and not enough memory to decode one, so committed tiles are
the only form in which a chart reaches a browser there. After adding or
updating a chart:

    make charts                     # or: make charts MAXZOOM=11
    git add data/charts/tiles

Each zoom level has four times the tiles of the one below, so the top one
or two are most of the bytes. What is committed is the sectional to z10 and
the terminal area charts to z12 -- 33 MB for the Bay Area set, against
93 MB for the full pyramid. MAXZOOM sets the cap, and MapLibre overzooms
past the top level rather than showing nothing, so a capped chart goes soft
rather than blank.

To trade storage back for sharpness:

    make charts MAXZOOM=11      # or leave MAXZOOM off for native zoom
