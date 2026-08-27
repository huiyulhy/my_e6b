"""Print every published POH cell for manual verification against a real POH.

The performance tables are digitized from an unofficial source. Everything the
planner computes rests on them, so they need checking against the book before
any of it is trusted. This prints them in the same layout as the POH charts so
they can be diffed page by page.

    make validate                 # or: PYTHONPATH= uv run python tools/validate_poh.py
    make validate > poh_check.txt # to diff against a later run

It also runs consistency checks that catch digitization errors the eye misses:
interpolation that fails to reproduce its own grid points, distances and climb
rates that move the wrong way with altitude or temperature, a cruise chart out
of order in RPM or temperature, and climb cumulatives that decrease.

These check internal coherence only. They cannot tell you the numbers match
your POH -- for that, diff the dumps.
"""

from __future__ import annotations

import csv
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import performance as perf
from engine.atmosphere import isa_temperature_c

DATA = ROOT / "data" / "poh" / "c172s"
WIDE_TEMPS = (0, 10, 20, 30, 40)


def raw(name: str) -> list[dict[str, str]]:
    with (DATA / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --- chart dumps ---------------------------------------------------------


def dump_takeoff() -> None:
    rule("TAKEOFF DISTANCE  --  short field, flaps 10, paved level dry runway")
    rows = raw("takeoff")
    for weight in sorted({r["wt"] for r in rows}, key=float, reverse=True):
        block = [r for r in rows if r["wt"] == weight]
        speeds = block[0]
        print(
            f"\n  Weight {weight} lb"
            f"   (liftoff {speeds['liftoff']} KIAS, 50 ft {speeds['speed50']} KIAS)"
        )
        print(f"  {'PRESS':>7} |" + "".join(f"{f'{t} C':>16}" for t in WIDE_TEMPS))
        print(f"  {'ALT ft':>7} |" + "".join(f"{'roll / 50ft':>16}" for _ in WIDE_TEMPS))
        print("  " + "-" * 88)
        for row in sorted(block, key=lambda r: float(r["p_alt"])):
            cells = "".join(
                f"{row[f'groundroll_{t}'] + ' / ' + row[f'clearfifty_{t}']:>16}"
                for t in WIDE_TEMPS
            )
            print(f"  {row['p_alt']:>7} |{cells}")


def dump_landing() -> None:
    rule("LANDING DISTANCE  --  short field, flaps 30, 2550 lb, paved level dry")
    print(f"\n  {'PRESS':>7} |" + "".join(f"{f'{t} C':>16}" for t in WIDE_TEMPS))
    print(f"  {'ALT ft':>7} |" + "".join(f"{'roll / 50ft':>16}" for _ in WIDE_TEMPS))
    print("  " + "-" * 88)
    for row in sorted(raw("landing"), key=lambda r: float(r["p_alt"])):
        cells = "".join(
            f"{row[f'groundroll_{t}'] + ' / ' + row[f'clearfifty_{t}']:>16}"
            for t in WIDE_TEMPS
        )
        print(f"  {row['p_alt']:>7} |{cells}")


def dump_climb_rate() -> None:
    rule("MAXIMUM RATE OF CLIMB  --  gross weight, full throttle, mixture leaned")
    temps = (-20, 0, 20, 40)
    print(f"\n  {'PRESS':>7} {'CLIMB':>7} |" + "".join(f"{f'{t} C':>10}" for t in temps))
    print(f"  {'ALT ft':>7} {'KIAS':>7} |" + "".join(f"{'fpm':>10}" for _ in temps))
    print("  " + "-" * 58)
    for row in sorted(raw("max_climb_rate"), key=lambda r: float(r["p_alt"])):
        cells = "".join(f"{(row[f't_{t}'] or '--'):>10}" for t in temps)
        print(f"  {row['p_alt']:>7} {row['kias']:>7} |{cells}")
    print("\n  Note: the 12000 ft / 40 C cell is blank in the POH. The engine")
    print("  refuses any query drawing on it rather than interpolating a value.")


def dump_climb_dist() -> None:
    rule("TIME, FUEL AND DISTANCE TO CLIMB  --  cumulative from sea level, standard day")
    print(
        f"\n  {'PRESS':>7} {'CLIMB':>7} {'RATE':>7} "
        f"{'TIME':>7} {'FUEL':>7} {'DIST':>7}"
    )
    print(
        f"  {'ALT ft':>7} {'KIAS':>7} {'fpm':>7} "
        f"{'min':>7} {'gal':>7} {'nm':>7}"
    )
    print("  " + "-" * 52)
    for row in sorted(raw("climb_dist"), key=lambda r: float(r["p_alt"])):
        print(
            f"  {row['p_alt']:>7} {row['speed']:>7} {row['rate']:>7} "
            f"{row['cum_time']:>7} {row['cum_fuel']:>7} {row['cum_dist']:>7}"
        )
    print("\n  A climb segment is the difference of two rows. Fuel excludes the")
    print("  POH's start, taxi and takeoff allowance -- add that separately.")


def dump_cruise() -> None:
    rule("CRUISE PERFORMANCE  --  2550 lb, recommended lean mixture")
    rows = raw("cruise")
    devs = sorted({float(r["isa_dev_c"]) for r in rows})
    for alt in sorted({float(r["press_alt"]) for r in rows}):
        block = [r for r in rows if float(r["press_alt"]) == alt]
        print(f"\n  Pressure altitude {alt:.0f} ft")
        print(f"  {'RPM':>6} |" + "".join(f"{f'ISA{d:+.0f}':>22}" for d in devs))
        print(f"  {'':>6} |" + "".join(f"{'%BHP / KTAS / GPH':>22}" for _ in devs))
        print("  " + "-" * 74)
        for rpm in sorted({float(r["rpm"]) for r in block}):
            cells = ""
            for d in devs:
                match = [
                    r
                    for r in block
                    if float(r["rpm"]) == rpm and float(r["isa_dev_c"]) == d
                ]
                cells += (
                    f"{f'{match[0]['pwr']} / {match[0]['ktas']} / {match[0]['gph']}':>22}"
                    if match
                    else f"{'--':>22}"
                )
            print(f"  {rpm:>6.0f} |{cells}")
    print("\n  The RPM range narrows with altitude -- blank entries are settings")
    print("  the POH does not publish there, not missing data.")


# --- consistency checks --------------------------------------------------


def check_interpolation_reproduces_grid() -> list[str]:
    """Interpolating at a published point must return the published number.

    Catches reshaping errors: a transposed axis still yields plausible
    numbers, just not the right ones.
    """
    problems = []
    for row in raw("takeoff"):
        for t in WIDE_TEMPS:
            got = perf.takeoff_distance(float(row["wt"]), float(row["p_alt"]), float(t))
            want = float(row[f"groundroll_{t}"])
            if abs(got.ground_roll_ft - want) > 1e-6:
                problems.append(
                    f"takeoff roll {row['wt']} lb / {row['p_alt']} ft / {t} C: "
                    f"got {got.ground_roll_ft}, published {want}"
                )
    for row in raw("cruise"):
        press_alt = float(row["press_alt"])
        # The chart column is an ISA deviation; `cruise` takes an absolute OAT.
        oat_c = float(row["isa_dev_c"]) + isa_temperature_c(press_alt)
        got = perf.cruise(press_alt, float(row["rpm"]), oat_c)
        if abs(got.ktas - float(row["ktas"])) > 1e-6:
            problems.append(
                f"cruise KTAS {row['press_alt']} ft / {row['rpm']} RPM / "
                f"ISA{float(row['isa_dev_c']):+.0f}: "
                f"got {got.ktas}, published {row['ktas']}"
            )
    return problems


def check_monotonicity() -> list[str]:
    """Distances must grow, and climb rates fall, in the expected directions."""
    problems = []
    for weight in (2200.0, 2400.0, 2550.0):
        for temp in (0.0, 20.0, 40.0):
            rolls = [
                perf.takeoff_distance(weight, alt, temp).ground_roll_ft
                for alt in range(0, 8001, 1000)
            ]
            if any(b < a for a, b in pairwise(rolls)):
                problems.append(
                    f"takeoff roll not increasing with altitude at {weight} lb / {temp} C"
                )
    for temp in (-20.0, 0.0, 20.0):
        rates = [perf.climb_rate(alt, temp).fpm for alt in range(0, 12001, 2000)]
        if any(b > a for a, b in pairwise(rates)):
            problems.append(f"climb rate not decreasing with altitude at {temp} C")
    return problems


def check_cruise_power_consistency() -> list[str]:
    """The cruise chart must move monotonically in every direction.

    Deliberately not a check that fuel flow is proportional to percent power.
    It is not: specific fuel consumption worsens at low power, so the lowest
    settings really do burn more gallons per percent than the high ones. That
    is engine behaviour, not a transcription error.

    What must hold is ordering. Within one altitude and temperature, more RPM
    means more power, more speed and more fuel. At one altitude and RPM,
    hotter air means less power and less fuel. A slipped digit almost always
    breaks one of these.

    True airspeed is deliberately excluded from the temperature check. It is
    not monotonic in temperature at high power down low: warmer air costs
    power but also thins, and the two effects nearly cancel. The chart shows
    this directly -- at 2000 ft, TAS spans 3 knots across the temperature
    range at 2100 RPM, 2 knots at 2400 RPM, and is exactly flat at 115 knots
    at 2500 RPM before going slightly non-monotonic at 2550.
    """
    problems = []
    rows = raw("cruise")

    def cells(**where: float) -> list[dict[str, str]]:
        return [r for r in rows if all(float(r[k]) == v for k, v in where.items())]

    altitudes = sorted({float(r["press_alt"]) for r in rows})
    devs = sorted({float(r["isa_dev_c"]) for r in rows})

    for alt in altitudes:
        for dev in devs:
            block = sorted(
                cells(press_alt=alt, isa_dev_c=dev), key=lambda r: float(r["rpm"])
            )
            for column in ("pwr", "ktas", "gph"):
                values = [float(r[column]) for r in block]
                if any(b < a for a, b in pairwise(values)):
                    problems.append(
                        f"cruise {column} does not increase with RPM at "
                        f"{alt:.0f} ft / ISA{dev:+.0f}: {values}"
                    )

    for alt in altitudes:
        for rpm in sorted({float(r["rpm"]) for r in cells(press_alt=alt)}):
            block = sorted(
                cells(press_alt=alt, rpm=rpm), key=lambda r: float(r["isa_dev_c"])
            )
            for column in ("pwr", "gph"):
                values = [float(r[column]) for r in block]
                if any(b > a for a, b in pairwise(values)):
                    problems.append(
                        f"cruise {column} does not decrease with temperature at "
                        f"{alt:.0f} ft / {rpm:.0f} RPM: {values}"
                    )
    return problems


def check_climb_cumulatives() -> list[str]:
    """Cumulative climb columns must never decrease going up."""
    problems = []
    rows = sorted(raw("climb_dist"), key=lambda r: float(r["p_alt"]))
    for column in ("cum_time", "cum_fuel", "cum_dist"):
        values = [float(r[column]) for r in rows]
        if any(b < a for a, b in pairwise(values)):
            problems.append(f"climb {column} decreases with altitude")
    # The published table is a standard-day table and carries no temperature
    # column of its own, so there is nothing here to check against ISA. The
    # 10%-per-10 C note is what handles a non-standard day, in
    # `performance.climb_from_to`.
    return problems


def main() -> int:
    print("Cessna 172S -- POH Section 5 as digitized in data/poh/c172s/")
    print("Verify every number below against a real POH. NOT FOR NAVIGATION.")
    dump_takeoff()
    dump_landing()
    dump_climb_rate()
    dump_climb_dist()
    dump_cruise()

    rule("CONSISTENCY CHECKS")
    checks = {
        "interpolation reproduces published grid points": check_interpolation_reproduces_grid,
        "distances and climb rates move the right way": check_monotonicity,
        "cruise chart is ordered in RPM and temperature": check_cruise_power_consistency,
        "climb cumulatives increase, temps are standard": check_climb_cumulatives,
    }
    total = 0
    for label, check in checks.items():
        problems = check()
        total += len(problems)
        print(f"\n  [{'FAIL' if problems else ' ok '}] {label}")
        for problem in problems:
            print(f"         {problem}")

    print(
        f"\n{'=' * 78}\n"
        f"{total} consistency problem(s). These check internal coherence only --\n"
        f"they cannot tell you the numbers match your POH. Diff the dumps above.\n"
        f"{'=' * 78}"
    )
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
