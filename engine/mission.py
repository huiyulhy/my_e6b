"""A mission: what the pilot decided, kept apart from what was worked out.

The navlog is always derived. What is worth keeping between the evening before
and the morning of the flight is the *intent* it was derived from: the fixes,
what each leg is asked to do about altitude (its events), the edits typed on
each leg, and the flight's own settings. Load that back, fetch the morning's
weather, and the planner lays the tops and bottoms of climb out again where
the new wind puts them -- nothing has to be typed twice.

So a mission holds a plan request with everything *observed* taken out: the
forecasts, each field's reported weather, and the points the planner inserted.
Those are fetched or derived again every time, and a mission that carried them
would plan tomorrow's flight in yesterday's air.

Beside it travels a **snapshot** of the last solve -- where each TOC and TOD
fell, the wind on each row, the totals. It is never planned from. It is what
the morning's plan is compared against, so the pilot can see what the new
weather changed.

Both are plain JSON, versioned by `SCHEMA`, and carried inside the exported KML
(see `engine.kml`) so the one file a pilot already keeps is the mission.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from engine.navlog import Navlog

__all__ = [
    "SCHEMA",
    "MissionError",
    "mission_json",
    "parse_mission",
    "parse_snapshot",
    "snapshot_json",
]

SCHEMA = 1

# Plan-request keys that describe the day rather than the plan.
_OBSERVED_REQUEST_KEYS = frozenset({"forecasts", "overrides"})

# Waypoint keys that are a field's reported weather, fetched again each time.
_OBSERVED_WAYPOINT_KEYS = frozenset(
    {
        "altimeter_inhg",
        "oat_c",
        "wind_from_deg",
        "wind_speed_kt",
        "gust_kt",
        "visibility_sm",
        "ceiling_ft_agl",
        "ceiling_cover",
        "sky_reported",
        "weather_reported",
    }
)

# A mission is a route and a few dozen edits. Anything this large is not one.
_MAX_CHARS = 1_000_000


class MissionError(ValueError):
    """The mission in a file cannot be read by this version."""


def mission_json(plan: dict[str, Any]) -> str:
    """The mission in a plan request, as JSON.

    `plan` is the request as a plain dict, in the shape `/api/plan` takes. The
    planner's own points go, and so does everything observed; the row-indexed
    `overrides` go too, because a row index means nothing once the rows have
    been laid out again -- edits worth keeping are the ones filed under a leg.
    """
    kept = {k: v for k, v in plan.items() if k not in _OBSERVED_REQUEST_KEYS}
    kept["waypoints"] = [
        {k: v for k, v in waypoint.items() if k not in _OBSERVED_WAYPOINT_KEYS}
        for waypoint in plan.get("waypoints", [])
        if not waypoint.get("generated")
    ]
    return json.dumps({"schema": SCHEMA, "plan": kept}, separators=(",", ":"), default=str)


def parse_mission(text: str) -> dict[str, Any]:
    """The plan request inside a mission, or `MissionError`.

    Only the envelope is checked here. The request itself is validated by
    whoever turns it into a plan, the same as one typed into the page.
    """
    document = _load(text, "mission")
    plan = document.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("waypoints"), list):
        raise MissionError("the mission in this file has no route")
    return plan


def snapshot_json(navlog: Navlog, *, solved_at: datetime | None = None) -> str:
    """The last solve, as JSON: enough to say what the next one changed."""
    rows = [
        {
            "from": leg.from_name,
            "to": leg.to_name,
            "segment_key": leg.segment_key,
            "phase": leg.phase,
            "start_role": leg.start_role,
            "end_role": leg.end_role,
            "to_lat": round(leg.to_position.lat, 6),
            "to_lon": round(leg.to_position.lon, 6),
            "cumulative_distance_nm": round(leg.cumulative_distance_nm, 2),
            "exit_altitude_ft": _rounded(leg.exit_altitude_ft),
            "wind_from_deg": round(leg.wind_from_deg),
            "wind_speed_kt": round(leg.wind_speed_kt),
            "wind_typed": "wind_from_deg" in leg.overridden
            or "wind_speed_kt" in leg.overridden,
            "ground_speed_kt": round(leg.ground_speed_kt, 1),
            "ete_min": round(leg.ete_min, 2),
            "fuel_gal": round(leg.fuel_gal, 2),
        }
        for leg in navlog.legs
        if leg.covers_ground
    ]
    return json.dumps(
        {
            "schema": SCHEMA,
            "solved_at": (solved_at or datetime.now(UTC)).isoformat(timespec="minutes"),
            "total_time_min": round(navlog.total_time_min, 2),
            "total_fuel_gal": round(navlog.total_fuel_gal, 2),
            "total_distance_nm": round(navlog.total_distance_nm, 2),
            "rows": rows,
        },
        separators=(",", ":"),
    )


def parse_snapshot(text: str) -> dict[str, Any]:
    """The last solve stored beside a mission, or `MissionError`."""
    document = _load(text, "snapshot")
    if not isinstance(document.get("rows"), list):
        raise MissionError("the plan snapshot in this file has no rows")
    return document


def _load(text: str, what: str) -> dict[str, Any]:
    if len(text) > _MAX_CHARS:
        raise MissionError(f"the {what} in this file is too large")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MissionError(f"the {what} in this file is not valid JSON: {exc}") from None
    if not isinstance(document, dict):
        raise MissionError(f"the {what} in this file is not an object")
    schema = document.get("schema")
    if schema != SCHEMA:
        raise MissionError(
            f"the {what} in this file is version {schema!r}; this planner reads "
            f"version {SCHEMA}"
        )
    return document


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value)
