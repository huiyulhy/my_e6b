"""Drive the UI in a real browser and screenshot it.

    make ui-shot          # requires `make serve` running on port 8137

Exists because the map is the one part of this project that unit tests cannot
reach: MapLibre needs WebGL, and a broken layer style or a typo'd colour fails
silently at runtime rather than at import. This adds a route, declares its
leg in the default user-driven mode, hands it back to the planner, and saves
images plus any console errors.

Uses the system Chrome via Playwright's `channel="chrome"` so no extra browser
download is needed. Headless Chrome has no GPU, so software WebGL is forced --
without it MapLibre never fires its `load` event and the map stays blank.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8137"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/e6b-ui")

# Force software WebGL: headless Chrome has no GPU, and MapLibre silently
# never finishes loading without a working GL context.
CHROME_FLAGS = [
    "--enable-unsafe-swiftshader",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--no-sandbox",
]

ROUTE = ["KSQL", "KMRY"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=CHROME_FLAGS)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on(
            "console",
            lambda m: problems.append(f"console.{m.type}: {m.text}")
            if m.type in ("error", "warning")
            else None,
        )
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))

        page.goto(BASE, wait_until="networkidle")
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('Ready')",
            timeout=30000,
        )
        print("map loaded")
        page.screenshot(path=OUT / "01-empty.png")

        for ident in ROUTE:
            page.fill("#search", ident)
            page.wait_for_selector("#results button", timeout=10000)
            page.click("#results button")
            page.wait_for_timeout(400)
            print(f"added {ident}")

        # User-driven is the default mode, so a freshly added route is
        # undeclared and the plan refuses until the pilot says what the leg
        # does. That prompt is the first thing a new user meets, so it is the
        # first thing checked.
        page.wait_for_timeout(1000)
        empty = page.locator("#navlog-empty")
        if empty.is_visible():
            print(f"\nundeclared route says: {empty.inner_text()}")
        else:
            problems.append("undeclared route planned without a segment type")
        page.screenshot(path=OUT / "02-undeclared.png")

        segment = page.locator("#waypoints select.segment").last
        segment.select_option("climb")
        page.wait_for_selector("#navlog tbody tr", timeout=15000)
        print(f"user-driven navlog rendered with "
              f"{page.locator('#navlog tbody tr').count()} legs")
        page.screenshot(path=OUT / "03-user-driven.png")

        # The rest exercises the planner. Handing the leg back to it needs both
        # the mode switch and the leg's own type cleared -- a declared leg stays
        # declared in either mode.
        page.click("#mode-auto")
        page.wait_for_timeout(400)
        page.locator("#waypoints select.segment").last.select_option("automatic")

        page.wait_for_selector("#navlog tbody tr", timeout=15000)
        rows = page.locator("#navlog tbody tr").count()
        print(f"navlog rendered with {rows} legs")
        page.wait_for_timeout(1200)
        page.screenshot(path=OUT / "04-route.png")

        summary = page.locator("#summary").inner_text()
        print("\nsummary:\n  " + summary.replace("\n", "\n  "))

        # A wind strong enough to matter, typed on the first leg of the
        # navlog, to prove a row edit drives a replan.
        wind_dir = page.locator("#navlog tbody tr td.wind-dir input").first
        wind_dir.fill("300")
        wind_dir.dispatch_event("change")
        page.wait_for_timeout(800)
        wind_speed = page.locator("#navlog tbody tr td.wind-speed input").first
        wind_speed.fill("25")
        wind_speed.dispatch_event("change")
        page.wait_for_timeout(1200)
        page.screenshot(path=OUT / "05-wind.png")
        print("\nwith wind:\n  " + page.locator("#summary").inner_text().replace("\n", "\n  "))

        # An altitude the POH does not publish, to prove refusals surface.
        page.fill("#altitude", "13500")
        page.dispatch_event("#altitude", "change")
        page.wait_for_timeout(1000)
        if empty.is_visible():
            print(f"\nrefusal shown to user: {empty.inner_text()}")
        page.screenshot(path=OUT / "06-refusal.png")

        browser.close()

    print(f"\nscreenshots in {OUT}")
    if problems:
        print(f"\n{len(problems)} console problem(s):")
        for problem in dict.fromkeys(problems):
            print(f"  {problem}")
        return 1
    print("no console errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
