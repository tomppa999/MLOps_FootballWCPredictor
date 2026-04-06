"""Utilities for inspecting fixture completion status from API-Football data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Statuses where a match is actively being played
IN_PROGRESS_STATUSES: frozenset[str] = frozenset({
    "1H",   # First Half
    "HT",   # Half Time
    "2H",   # Second Half
    "ET",   # Extra Time
    "BT",   # Break Time (between ET halves)
    "P",    # Penalty In Progress
    "SUSP", # Suspended (temporary)
    "INT",  # Interrupted (temporary)
    "LIVE", # Live (generic)
})

# Statuses where the result is final and will not change
FINISHED_STATUSES: frozenset[str] = frozenset({
    "FT",        # Full Time
    "AET",       # After Extra Time
    "PEN",       # After Penalties
    "AWD",       # Awarded (technical loss)
    "Abandoned", # Abandoned
})

# Statuses where the match will not be played or has not started yet
CANCELLED_OR_SCHEDULED_STATUSES: frozenset[str] = frozenset({
    "CANC", "Canc", # Cancelled
    "PST",          # Postponed
    "NS",           # Not Started
    "TBD",          # Time To Be Defined
    "WO",           # Walkover
})


def _get_status_short(fixture: dict[str, Any]) -> str:
    return fixture.get("fixture", {}).get("status", {}).get("short", "")


def check_fixtures_settled(fixtures_path: Path) -> tuple[bool, dict[str, Any]]:
    """Check whether all fixtures in a fixtures.json are settled (not in-progress).

    A fixture is considered settled if it is finished, cancelled, postponed,
    or not yet started. Any in-progress status (1H, HT, 2H, ET, etc.) causes
    this function to return False.

    Returns:
        (settled, summary) where settled is True iff no fixture is in-progress,
        and summary contains counts and the list of in-progress fixture IDs.
    """
    with fixtures_path.open(encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("response", [])
    counts: dict[str, int] = {
        "finished": 0,
        "cancelled_or_scheduled": 0,
        "in_progress": 0,
        "unknown": 0,
    }
    in_progress_ids: list[int] = []

    for fixture in fixtures:
        status = _get_status_short(fixture)
        fixture_id = fixture.get("fixture", {}).get("id")

        if status in IN_PROGRESS_STATUSES:
            counts["in_progress"] += 1
            if fixture_id is not None:
                in_progress_ids.append(fixture_id)
        elif status in FINISHED_STATUSES:
            counts["finished"] += 1
        elif status in CANCELLED_OR_SCHEDULED_STATUSES:
            counts["cancelled_or_scheduled"] += 1
        else:
            counts["unknown"] += 1

    summary: dict[str, Any] = {
        "total": len(fixtures),
        **counts,
        "in_progress_fixture_ids": in_progress_ids,
        "fixtures_path": str(fixtures_path),
    }
    settled = counts["in_progress"] == 0
    return settled, summary


def find_latest_fixtures_file(fixtures_dir: Path) -> Path | None:
    """Return the fixtures.json from the most recently named window directory.

    Window directories are named YYYY-MM-DD_YYYY-MM-DD, so lexicographic sort
    gives chronological order.
    """
    if not fixtures_dir.exists():
        return None
    windows = sorted(
        [d for d in fixtures_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    for window_dir in windows:
        candidate = window_dir / "fixtures.json"
        if candidate.exists():
            return candidate
    return None
