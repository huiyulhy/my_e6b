# Cessna 172S POH Section 5 data

Verified against C172S POH: https://www.befa.org/wp-content/uploads/2019/12/POH-Cessna-172S.pdf

Tables verified:
[X] Takeoff distance - 2550 lbs
[X] Takeoff distance - 2400 lbs
[X] Takeoff distance - 2200 lbs
[X] Climb distance
[X] Climb rate
[X] Landing distance
[X] Cruise performance (checked first column)

CSV source: <https://jhchou.github.io/c172s_performance/data/>
Modified to match POH for our codebase

## What is here
| File | Shape | Notes |
|---|---|---|
| `takeoff.csv` | weight × pressure altitude × temp | Full 3-D grid: 3 weights (2200/2400/2550 lb) × 9 altitudes (0–8000 ft) × 5 temps (0–40 °C). Temps are encoded in the *column names*. Short field, flaps 10. |
| `landing.csv` | pressure altitude × temp | 2550 lb only — the POH publishes landing distance at max weight only. Short field, flaps 30. |
| `max_climb_rate.csv` | pressure altitude × temp | 7 altitudes (0–12000 ft) × 4 temps (−20 to 40 °C). **Ragged:** the 12000 ft / 40 °C cell is blank in the POH, because the airplane will not climb there. Kept blank; the engine refuses queries that touch it. |
| `climb_dist.csv` | pressure altitude | Cumulative time/fuel/distance from sea level at standard temperature, plus climb speed. A climb segment's time and fuel are the difference of the cumulative columns interpolated at its two altitudes (not rounded out to the printed rows), and its speed is the average of the speed column at them; the table is printed at 2550 lb only and is used as printed at every weight; the distance column is digitized for validation but not used -- ground distance is flown against the wind. |
| `cruise.csv` | (altitude, RPM) × ISA deviation | **Ragged:** usable RPM narrows with altitude — 2100 RPM exists only at 2000–4000 ft, 2700 RPM only at 8000–10000 ft. The deviation axis is always full (−20/0/+20). |

### The cruise table's temperature axis is an ISA deviation, not an absolute OAT

The POH prints the cruise chart's three columns as **"20 °C BELOW STANDARD / STANDARD /
20 °C ABOVE STANDARD"**. The upstream CSV encoded those headings as the absolute
temperatures −5 / 15 / 35 °C, which are ISA−20 / ISA / ISA+20 *at sea level only*. The
column is stored here as `isa_dev_c` (−20 / 0 / +20) so it cannot be misread.

That the columns are ISA-relative is evident in the data itself: the same three values
appear at every altitude, and %BHP falls smoothly down each column (69 → 65 → 62 → 58 → 55
→ 52 at 2400 RPM). Read as absolute temperatures, one column would drift from ISA−16 at
2000 ft to ISA+4 at 12000 ft and could not produce a clean altitude-only progression.

`engine/performance.cruise` still takes an absolute OAT — every caller has one — and
converts internally at the queried pressure altitude.

## What is not here
- **Descent performance — the POH publishes none.** `engine/navlog.py` models descent as a
  pilot technique instead: a constant `descent_rate_fpm` (500) at `descent_tas_kt` (90 KTAS),
  with fuel charged at the cruise flow. That last part is deliberately conservative — a real
  172 descending at partial power burns less. These are assumptions, not book numbers.
- Crosswind component chart (Section 5)
- Airspeed calibration table (Section 5)
- Glide performance
- Weight and balance envelope (Section 6)

Airframe constants currently hardcoded in `engine/performance.py` and needing the same
verification: max gross 2550 lb, fuel 53 gal total / 50 usable, best glide 68 KIAS,
glide ratio 1.5 nm per 1000 ft.

## Corrections implemented:
`engine/performance.py` applies these corrections from the notes printed beneath the charts.

- Takeoff/landing: decrease distance 10% per 9 kt headwind; increase 10% per 2 kt tailwind.
- Takeoff on dry grass: add 15% of the ground roll.
- Landing on dry grass: add 45% of the ground roll.
- If landing with flaps up, increase the approach speed by 9 KIAS and 35% longer distances
- Climb: increase time, fuel and distance 10% per 10 °C above standard. Applied at the midpoint of each
  1000 ft band of the climb, and in both directions (a cold day is credited), floored at half the
  published figure.
- Add 1.4 gallons for engine start, and takeoff allowance
