"""Daily pipeline trigger.

Checks whether ELO and/or API-Football data have been updated since the last
run. If at least one source is fresh and all fixtures in the lookback window
are settled (not in-progress), the full ingestion → silver → gold pipeline is
executed and versioned via DVC.

Intentionally pandas-free so this module can be imported and tested in any
Python environment without requiring numpy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import requests

from src.ingestion.fixture_status import check_fixtures_settled, find_latest_fixtures_file

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (relative to project root; trigger must run from project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_FILE = _PROJECT_ROOT / "data" / "mappings" / "team_mapping_master_merged.csv"
_RAW_DIR = Path("data/raw")
_ELO_RAW_DIR = _RAW_DIR / "elo"
_ELO_MANIFEST_FILE = _ELO_RAW_DIR / "elo_manifest.csv"
_API_FOOTBALL_RAW_DIR = _RAW_DIR / "api_football"

LOOKBACK_DAYS = 2
_ELO_TSV_URL = "https://eloratings.net/{slug}.tsv"
_REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Stdlib-only ELO helpers (no pandas / numpy dependency)
# ---------------------------------------------------------------------------

def _load_manifest_hashes(manifest_file: Path) -> dict[str, str]:
    """Return {elo_slug: sha256} from the existing ELO manifest CSV."""
    if not manifest_file.exists():
        return {}
    with manifest_file.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {
            row["elo_slug"]: row["file_sha256"]
            for row in reader
            if row.get("file_sha256")
        }


def _load_slugs(mapping_file: Path) -> list[str]:
    """Return sorted unique ELO slugs from the team mapping CSV (stdlib only)."""
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    with mapping_file.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=csv.Sniffer().sniff(f.read(2048)))
        f.seek(0)
        reader = csv.DictReader(f)
        if "elo_slug" not in (reader.fieldnames or []):
            raise ValueError("Mapping file must contain an 'elo_slug' column.")
        slugs = {
            row["elo_slug"].strip()
            for row in reader
            if row.get("elo_slug", "").strip()
        }
    if not slugs:
        raise ValueError("No valid elo_slug values found.")
    return sorted(slugs)


def _download_to(url: str, target: Path, timeout: int = _REQUEST_TIMEOUT) -> tuple[bool, str]:
    """Download url to target path. Returns (success, message)."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        content = response.content
        if not content.strip():
            return False, "Empty response"
        target.write_bytes(content)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ELO freshness check
# ---------------------------------------------------------------------------

def check_elo_freshness(
    manifest_file: Path = _ELO_MANIFEST_FILE,
    mapping_file: Path = _MAPPING_FILE,
) -> bool:
    """Re-download ELO TSVs to a temp dir and return True if any SHA256 changed.

    Uses a temporary directory so live files are never overwritten just to
    check. The actual overwrite happens in run_elo_ingestion().
    """
    old_hashes = _load_manifest_hashes(manifest_file)
    if not old_hashes:
        log.warning("No existing ELO manifest found; treating ELO as fresh.")
        return True

    try:
        slugs = _load_slugs(mapping_file)
    except Exception as exc:
        log.error(f"Failed to load ELO slugs: {exc}")
        return False

    changed = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for slug in slugs:
            tmp_tsv = Path(tmpdir) / f"{slug}.tsv"
            success, msg = _download_to(_ELO_TSV_URL.format(slug=slug), tmp_tsv)
            if not success:
                log.warning(f"ELO temp download failed for {slug}: {msg}")
                continue
            if _sha256(tmp_tsv) != old_hashes.get(slug):
                log.info(f"ELO data changed: {slug}")
                changed += 1

    log.info(f"ELO freshness: {changed}/{len(slugs)} teams changed")
    return changed > 0


# ---------------------------------------------------------------------------
# API-Football freshness check
# ---------------------------------------------------------------------------

def _find_latest_run_manifest(runs_dir: Path) -> Path | None:
    """Return the most recently named run manifest JSON, or None."""
    if not runs_dir.exists():
        return None
    manifests = sorted(runs_dir.glob("*_run_manifest.json"), reverse=True)
    return manifests[0] if manifests else None


def check_api_football_freshness(
    runs_dir: Path = _API_FOOTBALL_RAW_DIR / "runs",
    fixtures_dir: Path = _API_FOOTBALL_RAW_DIR / "fixtures",
    lookback_days: int = LOOKBACK_DAYS,
) -> bool:
    """Run incremental ingestion and return True if new, finished fixtures exist.

    The ingestion is the check: we fetch the last `lookback_days` of data and
    inspect what came back. If the pipeline later runs, the already-downloaded
    data is used without a second API call.
    """
    result = subprocess.run(
        [
            sys.executable, "-m",
            "src.ingestion.download_api_football_national_matches",
            "--lookback-days", str(lookback_days),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"API-Football ingestion failed:\n{result.stderr}")
        return False

    manifest_path = _find_latest_run_manifest(runs_dir)
    if manifest_path is None:
        log.warning("No API-Football run manifest found after ingestion.")
        return False

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    kept = manifest.get("summary", {}).get("kept_fixture_count_total", 0)
    if kept == 0:
        log.info(f"API-Football: no fixtures in {lookback_days}-day window.")
        return False

    fixtures_file = find_latest_fixtures_file(fixtures_dir)
    if fixtures_file is None:
        log.warning("No fixtures.json found after ingestion.")
        return False

    settled, summary = check_fixtures_settled(fixtures_file)
    log.info(
        f"API-Football: kept={kept}, settled={settled}, "
        f"finished={summary['finished']}, in_progress={summary['in_progress']}"
    )
    return settled and summary["finished"] > 0


# ---------------------------------------------------------------------------
# Ingestion and pipeline execution
# ---------------------------------------------------------------------------

def run_elo_ingestion() -> None:
    """Re-download all ELO TSVs with overwrite to capture the updated data.

    Sets ELO_OVERWRITE=1 in the subprocess environment, which download_elo_tsvs
    reads at import time to enable overwrite mode.
    """
    result = subprocess.run(
        [sys.executable, "-m", "src.ingestion.download_elo_tsvs"],
        env={**os.environ, "ELO_OVERWRITE": "1"},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ELO ingestion failed:\n{result.stderr}")
    log.info("ELO ingestion complete.")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log.info(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def run_dvc_pipeline() -> None:
    """Update the DVC raw pointer, rebuild silver/gold, push data and commit."""
    _run(["dvc", "add", "data/raw"])
    _run(["dvc", "repro"])
    _run(["dvc", "push"])
    _run(["git", "add", "dvc.lock", "data/raw.dvc"])

    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
    if diff.returncode != 0:
        today = date.today().isoformat()
        _run(["git", "commit", "-m", f"data: auto-update pipeline {today}"])
        _run(["git", "push"])
        log.info("DVC pipeline committed and pushed.")
    else:
        log.info("No DVC changes to commit (data unchanged).")


VALID_MODES = ("auto", "inference_only")


def dispatch_training_or_inference(mode: str = "auto") -> None:
    """Decide whether to train, refit, or run inference only.

    - 'auto': three paths based on state —
        1. No production champion → run full pipeline (Experimental + QA + Deploy).
        2. Champion exists, Gold grew by >= RETRAIN_THRESHOLD rows → refit champion.
        3. Champion exists, delta below threshold → inference only.
    - 'inference_only': load frozen champion from MLflow, predict + simulate.

    All paths end with ``run_inference_and_simulation()``.
    """
    import mlflow  # noqa: PLC0415

    from src.inference.run import run_inference_and_simulation  # noqa: PLC0415
    from src.models.data_split import load_gold  # noqa: PLC0415
    from src.models.mlflow_utils import (  # noqa: PLC0415
        get_latest_production_run_id,
        setup_mlflow,
    )
    from src.models.pipeline import (  # noqa: PLC0415
        RETRAIN_THRESHOLD,
        run_champion_refit,
        run_full_pipeline,
    )

    setup_mlflow()

    df = load_gold()
    current_rows = len(df)
    log.info("Gold row count: %d", current_rows)

    if mode == "inference_only":
        log.info("Mode is inference_only — skipping retrain check.")
        run_inference_and_simulation()
        return

    prod_run_id = get_latest_production_run_id()
    if prod_run_id is None:
        log.info("No production champion found — running full pipeline.")
        run_full_pipeline(df)
        run_inference_and_simulation()
        return

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(prod_run_id).data
    last_rows = int(run_data.params.get("gold_row_count", "0"))
    delta = current_rows - last_rows
    log.info(
        "Gold delta: %d (current=%d, last=%d)",
        delta,
        current_rows,
        last_rows,
    )

    if delta >= RETRAIN_THRESHOLD:
        log.info("Refit threshold met — refitting champion on fresh data.")
        run_champion_refit(df)
        run_inference_and_simulation()
    else:
        log.info("Retrain threshold not met — running inference only.")
        run_inference_and_simulation()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(mode: str = "auto") -> int:
    from dotenv import load_dotenv  # noqa: PLC0415
    load_dotenv()

    if os.environ.get("DAGSHUB_USERNAME") and not os.environ.get("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN", "")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    if mode not in VALID_MODES:
        log.error(f"Invalid mode: {mode!r}. Must be one of {VALID_MODES}.")
        return 1

    log.info(f"=== Pipeline trigger start (mode={mode}) ===")

    log.info("Checking ELO freshness...")
    elo_fresh = check_elo_freshness()
    log.info(f"ELO fresh: {elo_fresh}")

    log.info("Checking API-Football freshness (runs incremental ingestion)...")
    api_fresh = check_api_football_freshness()
    log.info(f"API-Football fresh: {api_fresh}")

    if not (elo_fresh or api_fresh):
        log.info(f"No source has new data. elo_fresh={elo_fresh}, api_fresh={api_fresh}")
        return 0

    log.info(f"New data detected (elo={elo_fresh}, api={api_fresh}) — running pipeline.")

    if elo_fresh:
        run_elo_ingestion()

    run_dvc_pipeline()
    dispatch_training_or_inference(mode=mode)

    log.info("=== Pipeline trigger complete ===")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline trigger")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="auto",
        help="auto = retrain if enough new data, then infer; inference_only = frozen model",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(mode=args.mode))
