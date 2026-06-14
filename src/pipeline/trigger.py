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
import fcntl
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

# Safety floor: refuse to snapshot/push a collapsed raw dataset. When `dvc pull`
# fails to restore the historical raw in a fresh container, data/raw contains
# only the freshly-ingested lookback window (tens of files) instead of the full
# history (thousands). Pushing that would clobber the remote DVC pointer with a
# truncated dataset (as happened on 2026-06-08). Overridable via env for tests.
RAW_MIN_FILES = int(os.getenv("RAW_MIN_FILES", "1000"))

# Concurrency guard: prevents a second local invocation from running while the
# first is still active.  On Cloud Run, max-instances=1 (C.5) is the primary
# cross-execution guard; this lockfile covers within-host overlap (local dev /
# manual runs).  Each Cloud Run execution gets a fresh container filesystem so
# the lock does not persist between executions there.
_LOCK_FILE = _PROJECT_ROOT / "data" / ".trigger.lock"


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

    _, summary = check_fixtures_settled(fixtures_file)
    if summary["in_progress"] > 0:
        log.warning(
            f"API-Football: {summary['in_progress']} fixture(s) still in progress "
            f"(IDs: {summary['in_progress_fixture_ids']}); processing settled matches anyway."
        )
    log.info(
        f"API-Football: kept={kept}, finished={summary['finished']}, "
        f"in_progress={summary['in_progress']}"
    )
    return summary["finished"] > 0


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


def _count_files(root: Path) -> int:
    """Recursively count regular files under ``root`` (0 if it does not exist)."""
    if not root.exists():
        return 0
    return sum(len(files) for _, _, files in os.walk(root))


def run_dvc_pipeline() -> None:
    """Update the DVC raw pointer, rebuild silver/gold, push data and commit."""
    raw_files = _count_files(_RAW_DIR)
    if raw_files < RAW_MIN_FILES:
        raise RuntimeError(
            f"Refusing to update DVC: data/raw has {raw_files} files "
            f"(< RAW_MIN_FILES={RAW_MIN_FILES}). This indicates `dvc pull` did "
            f"not restore the historical raw; aborting before a truncated "
            f"dataset is snapshotted and pushed over the remote pointer."
        )
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


def _try_acquire_lock():
    """Non-blocking attempt to acquire the per-host trigger lockfile.

    Returns an open file handle (lock held) on success; the caller must close
    it to release the lock.  Returns None when the lock is already held.
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = open(_LOCK_FILE, "w")  # noqa: WPS515
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except (BlockingIOError, OSError):
        fh.close()
        return None


def _last_per_round_refit_matchday() -> str | None:
    """Return the matchday label stored on the champion_per_round run.

    Returns ``None`` when the ``champion_per_round`` alias does not exist yet
    (pre-A.10 / pre-first-freeze), indicating the pipeline should use the
    legacy delta-based refit path.  Returns ``"0"`` when the alias exists but
    no ``per_round_refit_matchday`` tag has been recorded (edge case: alias
    assigned manually without a tagged refit run).
    """
    import mlflow  # noqa: PLC0415

    from src.models.mlflow_utils import (  # noqa: PLC0415
        CHAMPION_ALIAS_PER_ROUND,
        PRODUCTION_MODEL_NAME,
        setup_mlflow,
    )

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(PRODUCTION_MODEL_NAME, CHAMPION_ALIAS_PER_ROUND)
    except mlflow.exceptions.MlflowException:
        return None
    try:
        return client.get_run(mv.run_id).data.tags.get("per_round_refit_matchday", "0")
    except Exception:
        log.warning("Failed to read run tags for champion_per_round — treating as first refit.")
        return "0"


def _last_per_round_completed_matchday() -> str:
    """Return the last-completed matchday label stored on the champion_per_round run.

    Used as the gate for the per-round refit: the refit fires when
    ``parse_wc_results``'s ``last_completed_matchday`` advances past this value.

    Returns ``"0"`` when the tag is absent (no round completed yet), the alias
    does not exist, or the run cannot be read — making the gate fire as soon
    as the first round fully completes.
    """
    import mlflow  # noqa: PLC0415

    from src.models.mlflow_utils import (  # noqa: PLC0415
        CHAMPION_ALIAS_PER_ROUND,
        PRODUCTION_MODEL_NAME,
        setup_mlflow,
    )

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(PRODUCTION_MODEL_NAME, CHAMPION_ALIAS_PER_ROUND)
    except mlflow.exceptions.MlflowException:
        return "0"
    try:
        return client.get_run(mv.run_id).data.tags.get(
            "per_round_last_completed_matchday", "0"
        )
    except Exception:
        log.warning(
            "Failed to read per_round_last_completed_matchday tag — treating as no round completed."
        )
        return "0"


VALID_MODES = ("auto", "inference_only")
CADENCE_MODES: tuple[str, ...] = ("frozen", "per_round")


def _safe_monitoring_step() -> None:
    """Best-effort monitoring hook. A failure here must not gate the pipeline."""
    try:
        from src.monitoring.monitor import run_monitoring_step  # noqa: PLC0415
        run_monitoring_step()
    except Exception:
        log.exception("Monitoring step failed; continuing pipeline.")


def _run_inference_for_all_modes() -> None:
    """Run inference + simulation once per cadence mode (B.2).

    No explicit seed is passed; each call derives a seed from the matchday
    label it resolves internally. Both modes see the same matchday and therefore
    produce the same seed, keeping frozen vs per_round comparisons free of MC
    noise within a cycle. The seed changes at each matchday boundary, preserving
    cross-snapshot independence for the RQ2 entropy trajectory.
    """
    from src.inference.run import run_inference_and_simulation  # noqa: PLC0415

    for mode in CADENCE_MODES:
        log.info("Running inference for cadence_mode=%s", mode)
        run_inference_and_simulation(cadence_mode=mode)


def _safe_shadow_refit(df) -> None:
    """Best-effort shadow refit. A failure here must not gate the pipeline."""
    try:
        from src.models.pipeline import run_shadow_refit  # noqa: PLC0415
        run_shadow_refit(df)
    except Exception:
        log.exception("Shadow refit failed; continuing pipeline.")


def _safe_per_round_refit(df, matchday: str, completed_matchday: str) -> None:
    """Best-effort per-round roster refit (B.3). Must not gate inference."""
    try:
        from src.models.pipeline import run_per_round_refit  # noqa: PLC0415
        run_per_round_refit(df, matchday=matchday, completed_matchday=completed_matchday)
    except Exception:
        log.exception(
            "Per-round refit failed (matchday=%s); inference continues.", matchday,
        )


def dispatch_training_or_inference(mode: str = "auto") -> None:
    """Decide whether to train, refit, or run inference only.

    - 'auto': three paths based on state —
        1. No production champion → run full pipeline (Experimental + QA + Deploy).
        2. Champion exists, ``champion_per_round`` alias assigned (post-A.10,
           WC mode) → per-round boundary-gated roster refit; frozen never
           refits during WC.
        3. Champion exists, no ``champion_per_round`` alias (pre-A.10) →
           legacy delta-based champion refit path.
    - 'inference_only': load frozen champion from MLflow, predict + simulate.

    All paths end with ``_run_inference_for_all_modes()`` (both cadence modes)
    followed by a best-effort monitoring step.
    """
    import mlflow  # noqa: PLC0415

    from src.models.data_split import load_gold  # noqa: PLC0415
    from src.models.mlflow_utils import (  # noqa: PLC0415
        get_latest_production_run_id,
        setup_mlflow,
    )
    from src.models.pipeline import (  # noqa: PLC0415
        run_champion_refit,
        run_full_pipeline,
    )

    # Default 1 so every new match triggers a refit (WC cadence).
    # Override with RETRAIN_THRESHOLD env var for testing / back-compat.
    retrain_threshold = int(os.getenv("RETRAIN_THRESHOLD", "1"))

    setup_mlflow()

    df = load_gold()
    current_rows = len(df)
    log.info("Gold row count: %d", current_rows)

    if mode == "inference_only":
        log.info("Mode is inference_only — skipping retrain check.")
        _run_inference_for_all_modes()
        _safe_monitoring_step()
        return

    prod_run_id = get_latest_production_run_id()
    if prod_run_id is None:
        log.info("No production champion found — running full pipeline.")
        run_full_pipeline(df)
        _safe_shadow_refit(df)
        _run_inference_for_all_modes()
        _safe_monitoring_step()
        return

    # B.3: WC mode — per-round boundary-gated refit (post-A.10).
    # Active once champion_per_round alias is assigned; frozen never refits.
    # Gate: fire when a round has FULLY completed (all scheduled fixtures
    # finished), not merely when the first game of the next round exists.
    # Uses the new per_round_last_completed_matchday tag so the gate is
    # independent of the legacy per_round_refit_matchday tag, which may carry
    # stale values from pre-fix runs (e.g. the premature MD1 refit of Jun 12).
    last_per_round_matchday = _last_per_round_refit_matchday()
    if last_per_round_matchday is not None:
        from src.inference.features import parse_wc_results  # noqa: PLC0415

        wc_data = parse_wc_results()
        next_matchday = str(wc_data["next_matchday"])
        last_completed = str(wc_data["last_completed_matchday"])
        last_tagged = _last_per_round_completed_matchday()

        if last_completed != "0" and last_completed != last_tagged and next_matchday != "Complete":
            log.info(
                "Round fully complete: %s → %s; per-round roster refit for matchday=%s.",
                last_tagged, last_completed, next_matchday,
            )
            _safe_per_round_refit(df, matchday=next_matchday, completed_matchday=last_completed)
        else:
            log.info(
                "No completed round advance "
                "(last_completed=%s, tagged=%s, next=%s) — inference only.",
                last_completed, last_tagged, next_matchday,
            )
        _run_inference_for_all_modes()
        _safe_monitoring_step()
        return

    # Pre-A.10 legacy path: delta-based champion refit.
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(prod_run_id).data
    last_rows = int(run_data.params.get("gold_row_count", "0"))
    delta = current_rows - last_rows
    log.info(
        "Gold delta: %d (current=%d, last=%d, threshold=%d)",
        delta,
        current_rows,
        last_rows,
        retrain_threshold,
    )

    if delta >= retrain_threshold:
        log.info("Refit threshold met — refitting champion on fresh data.")
        run_champion_refit(df)
        _safe_shadow_refit(df)
        _run_inference_for_all_modes()
        _safe_monitoring_step()
    else:
        log.info("Retrain threshold not met — running inference only.")
        _run_inference_for_all_modes()
        _safe_monitoring_step()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(mode: str = "auto") -> int:
    from dotenv import load_dotenv  # noqa: PLC0415
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    if mode not in VALID_MODES:
        log.error(f"Invalid mode: {mode!r}. Must be one of {VALID_MODES}.")
        return 1

    lock_fh = _try_acquire_lock()
    if lock_fh is None:
        log.info("Previous pipeline cycle still running — skipping this tick.")
        return 0

    try:
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
    finally:
        lock_fh.close()


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
