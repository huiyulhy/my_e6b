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

Tiles rendered from the charts land in two places, and only one of them
is committed:

  data/charts/tiles/     what `make charts` built. COMMITTED.
  .cache/chart-tiles/    what the dev server rendered as you panned past
                         the committed zoom cap. Ignored, delete any time.

The tiles are committed and the GeoTIFFs are not, which is backwards from
the usual rule and deliberate. A server deploying this has no GeoTIFFs and
not enough memory to decode one -- a sectional peaks around 410 MB -- so
committed tiles are the only form in which a chart reaches a browser there,
and they mean a deploy needs no build step at all. The GeoTIFFs are a
re-download away; the tiles are what the app ships with.

Only `make charts` writes the committed directory, so the two never mix.
After adding a chart or taking a new 56-day cycle:

    make charts MAXZOOM=12      # what is committed today
    git add data/charts/tiles

A chart whose source has not changed is skipped, so re-running is cheap. A
new edition is detected by its date and file size and re-rendered, which
matters: the FAA reissues under the same filename, and without that check
the old edition's pixels would survive under the new edition's dates.

Each zoom level has four times the tiles of the one below, so the top one
or two are most of the bytes. The sectionals are capped at z10 and the
terminal area charts at z12. MapLibre overzooms past the top level rather
than showing nothing, so a capped chart goes soft rather than blank.
