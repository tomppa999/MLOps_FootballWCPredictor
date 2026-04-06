import json
from pathlib import Path

import pytest

from src.ingestion.fixture_status import (
    FINISHED_STATUSES,
    IN_PROGRESS_STATUSES,
    check_fixtures_settled,
    find_latest_fixtures_file,
)


def _make_fixture(fixture_id: int, status_short: str) -> dict:
    return {"fixture": {"id": fixture_id, "status": {"short": status_short, "long": ""}}}


def _write_fixtures(path: Path, fixtures: list[dict]) -> Path:
    path.write_text(json.dumps({"response": fixtures}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# check_fixtures_settled
# ---------------------------------------------------------------------------

def test_all_finished_is_settled(tmp_path):
    path = _write_fixtures(tmp_path / "fixtures.json", [
        _make_fixture(1, "FT"),
        _make_fixture(2, "AET"),
        _make_fixture(3, "PEN"),
    ])
    settled, summary = check_fixtures_settled(path)
    assert settled is True
    assert summary["finished"] == 3
    assert summary["in_progress"] == 0
    assert summary["in_progress_fixture_ids"] == []


def test_in_progress_match_is_not_settled(tmp_path):
    path = _write_fixtures(tmp_path / "fixtures.json", [
        _make_fixture(1, "FT"),
        _make_fixture(2, "HT"),
    ])
    settled, summary = check_fixtures_settled(path)
    assert settled is False
    assert summary["in_progress"] == 1
    assert 2 in summary["in_progress_fixture_ids"]


def test_all_in_progress_statuses_block_settled(tmp_path):
    for status in IN_PROGRESS_STATUSES:
        path = _write_fixtures(tmp_path / f"fixtures_{status}.json", [
            _make_fixture(99, status),
        ])
        settled, _ = check_fixtures_settled(path)
        assert settled is False, f"Status '{status}' should block settled"


def test_cancelled_and_postponed_are_settled(tmp_path):
    path = _write_fixtures(tmp_path / "fixtures.json", [
        _make_fixture(1, "CANC"),
        _make_fixture(2, "PST"),
        _make_fixture(3, "FT"),
    ])
    settled, summary = check_fixtures_settled(path)
    assert settled is True
    assert summary["cancelled_or_scheduled"] == 2
    assert summary["finished"] == 1


def test_empty_fixture_list_is_settled(tmp_path):
    path = _write_fixtures(tmp_path / "fixtures.json", [])
    settled, summary = check_fixtures_settled(path)
    assert settled is True
    assert summary["total"] == 0


def test_summary_contains_expected_keys(tmp_path):
    path = _write_fixtures(tmp_path / "fixtures.json", [_make_fixture(1, "FT")])
    _, summary = check_fixtures_settled(path)
    for key in ("total", "finished", "cancelled_or_scheduled", "in_progress", "unknown",
                "in_progress_fixture_ids", "fixtures_path"):
        assert key in summary


def test_unknown_status_counted_as_unknown_not_blocking(tmp_path):
    path = _write_fixtures(tmp_path / "fixtures.json", [
        _make_fixture(1, "WEIRD"),
    ])
    settled, summary = check_fixtures_settled(path)
    assert settled is True
    assert summary["unknown"] == 1


def test_mixed_in_progress_reports_all_ids(tmp_path):
    path = _write_fixtures(tmp_path / "fixtures.json", [
        _make_fixture(10, "1H"),
        _make_fixture(11, "FT"),
        _make_fixture(12, "2H"),
    ])
    settled, summary = check_fixtures_settled(path)
    assert settled is False
    assert set(summary["in_progress_fixture_ids"]) == {10, 12}


# ---------------------------------------------------------------------------
# find_latest_fixtures_file
# ---------------------------------------------------------------------------

def test_find_latest_returns_most_recent_window(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    old = fixtures_dir / "2025-01-01_2025-01-02"
    new = fixtures_dir / "2025-03-30_2025-03-31"
    for d in [old, new]:
        d.mkdir(parents=True)
        (d / "fixtures.json").write_text('{"response": []}')

    result = find_latest_fixtures_file(fixtures_dir)
    assert result == new / "fixtures.json"


def test_find_latest_returns_none_when_empty(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    assert find_latest_fixtures_file(fixtures_dir) is None


def test_find_latest_skips_window_without_fixtures_json(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    no_file = fixtures_dir / "2025-03-31_2025-03-31"
    has_file = fixtures_dir / "2025-03-29_2025-03-29"
    no_file.mkdir(parents=True)
    has_file.mkdir(parents=True)
    (has_file / "fixtures.json").write_text('{"response": []}')

    result = find_latest_fixtures_file(fixtures_dir)
    assert result == has_file / "fixtures.json"
