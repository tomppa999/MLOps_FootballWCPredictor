"""Freeze the MLflow inputs of the D.1 reconstruction into DVC-tracked files.

This is the only step of the post-hoc analysis that needs DagsHub.  It runs
once (by the author) and writes ``data/reconstruction/inputs/``:

- ``logged_predictions/all_models.parquet`` / ``champion.parquet`` — the
  per-cycle logged lambdas the Strand 2 replay simulates from, restricted to
  :data:`~src.models.config.EXPERIMENT_MODELS` and to the columns
  ``simulate_tournament`` actually reads.
- ``models/<registry>/v<N>/`` — the pinned pyfunc directories Strands 1 and 5
  predict with, plus ``models/manifest.csv`` recording which registry version
  served which model in which cadence/regime.
- ``fixture_settlement.csv`` — when live first saw each WC fixture finished,
  walked out of the versioned Bronze (needs DVC, not MLflow).

Afterwards every strand runs offline; MLflow stays only as a fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.analysis.replay_common import (
    FIXTURE_SETTLEMENT_COLUMNS,
    FIXTURE_SETTLEMENT_PATH,
    FROZEN_REGIME,
    MODEL_MANIFEST_COLUMNS,
    MODEL_MANIFEST_PATH,
    PREDICTION_KINDS,
    RAW_FIXTURES_DVC_PATH,
    SNAPSHOT_PREDICTION_COLUMNS,
    _SNAPSHOT_ID_COLUMN,
    FinishedFixture,
    GoldCommit,
    build_gold_commit_index,
    load_finished_wc_fixtures,
    load_inference_cycles,
    snapshot_model_dir,
    snapshot_predictions_path,
)
from src.inference.features import FINISHED_STATUSES
from src.analysis.strand2_brackets import (
    _CHAMPION_PREDICTIONS_FILENAME,
    _PREDICTIONS_FILENAME,
    _load_artifact,
)
from src.models.config import EXPERIMENT_MODELS
from src.models.mlflow_utils import (
    PRODUCTION_MODEL_NAME,
    SHADOW_MODEL_NAME,
    setup_mlflow,
)

logger = logging.getLogger(__name__)

_ARTIFACT_FILENAMES: Final[dict[str, str]] = {
    "all_models": _PREDICTIONS_FILENAME,
    "champion": _CHAMPION_PREDICTIONS_FILENAME,
}


@dataclass(frozen=True)
class ModelSnapshotSpec:
    """One registry version to download, with the regime it served.

    ``model_name`` is deliberately absent: it is resolved from the backing
    run's ``model_name`` tag so a wrong assumption about which shadow version
    holds which model cannot silently propagate into the manifest.
    """

    registry_name: str
    version: int
    cadence_role: str
    regime: str


# Which registry versions were actually served, per cadence and regime.
# Sources: docs/notes/wc_live.md regime table (L564-580) for the per_round
# refit boundaries, and the pre-tournament fit for frozen (which never refits).
MODEL_SNAPSHOT_SPECS: Final[tuple[ModelSnapshotSpec, ...]] = (
    # Regime 1 — pre-tournament fit; what "frozen" served all tournament long.
    ModelSnapshotSpec(PRODUCTION_MODEL_NAME, 15, "frozen", FROZEN_REGIME),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 88, "frozen", FROZEN_REGIME),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 89, "frozen", FROZEN_REGIME),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 90, "frozen", FROZEN_REGIME),
    # Regime 5 — the R32 window (Jun 28 19:00 – Jul 4 01:30).
    ModelSnapshotSpec(PRODUCTION_MODEL_NAME, 19, "per_round", "R32"),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 104, "per_round", "R32"),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 105, "per_round", "R32"),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 106, "per_round", "R32"),
    # Regime 6 — the R16 window (Jul 4 17:00 – Jul 7 20:00).
    ModelSnapshotSpec(PRODUCTION_MODEL_NAME, 20, "per_round", "R16"),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 107, "per_round", "R16"),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 108, "per_round", "R16"),
    ModelSnapshotSpec(SHADOW_MODEL_NAME, 109, "per_round", "R16"),
)


# ---------------------------------------------------------------------------
# Logged predictions
# ---------------------------------------------------------------------------


def _arrow_schema(kind: str) -> pa.Schema:
    fields = [pa.field(_SNAPSHOT_ID_COLUMN, pa.string())]
    for col in SNAPSHOT_PREDICTION_COLUMNS[kind]:
        dtype = pa.float64() if col.startswith("lambda_") else pa.string()
        fields.append(pa.field(col, dtype))
    return pa.schema(fields)


def _prepare_rows(df: pd.DataFrame, run_id: str, kind: str) -> pd.DataFrame:
    """Restrict one logged artifact to the snapshot rows/columns, in order."""
    if kind == "all_models":
        df = df[df["model_name"].isin(EXPERIMENT_MODELS)]
    cols = list(SNAPSHOT_PREDICTION_COLUMNS[kind])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"run {run_id}: logged {kind} artifact missing {missing}")
    out = df[cols].copy()
    out.insert(0, _SNAPSHOT_ID_COLUMN, run_id)
    return out


def build_prediction_snapshot(
    *,
    cycles_path: Path | None = None,
    refresh: bool = False,
) -> dict[str, Path]:
    """Consolidate every cycle's logged predictions into two parquet files.

    Rows stay in logged order and grouped per run, which is what the offline
    reader relies on to slice a run without a groupby.  Missing artifacts (the
    two pre-tournament cycles that logged none) are skipped with a warning —
    the Strand 2 replay skips them too.
    """
    cycles = (
        load_inference_cycles(cycles_path) if cycles_path else load_inference_cycles()
    )
    run_ids = cycles["inference_run_id"].astype(str).tolist()
    logger.info("Snapshotting logged predictions for %d inference cycles", len(run_ids))

    paths: dict[str, Path] = {}
    for kind in PREDICTION_KINDS:
        out_path = snapshot_predictions_path(kind)
        if out_path.exists() and not refresh:
            logger.info("Skipping existing %s (pass --refresh to rebuild)", out_path)
            paths[kind] = out_path
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        schema = _arrow_schema(kind)
        written = 0
        skipped: list[str] = []
        staged = out_path.with_suffix(out_path.suffix + ".tmp")
        with pq.ParquetWriter(staged, schema, compression="snappy") as writer:
            for run_id in run_ids:
                df = _load_artifact(run_id, _ARTIFACT_FILENAMES[kind])
                if df is None:
                    skipped.append(run_id)
                    continue
                rows = _prepare_rows(df, run_id, kind)
                writer.write_table(pa.Table.from_pandas(rows, schema=schema, preserve_index=False))
                written += 1
        staged.replace(out_path)

        logger.info(
            "Wrote %s: %d runs (%.1f MB), %d without artifacts",
            out_path,
            written,
            out_path.stat().st_size / 1e6,
            len(skipped),
        )
        if skipped:
            logger.warning("No %s artifact for runs: %s", kind, skipped)
        paths[kind] = out_path

    return paths


# ---------------------------------------------------------------------------
# Pinned models
# ---------------------------------------------------------------------------


def _resolve_model_name(client: mlflow.tracking.MlflowClient, spec: ModelSnapshotSpec) -> tuple[str, str]:
    """Return ``(model_name, source_run_id)`` from the version's backing run."""
    mv = client.get_model_version(spec.registry_name, str(spec.version))
    tags = client.get_run(mv.run_id).data.tags
    model_name = tags.get("model_name")
    if not model_name:
        raise ValueError(
            f"{spec.registry_name} v{spec.version} (run {mv.run_id}) has no "
            "model_name tag — cannot pin it to a model",
        )
    return model_name, mv.run_id


def build_model_snapshot(*, refresh: bool = False) -> Path:
    """Download every pinned pyfunc directory and write ``models/manifest.csv``."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    rows: list[dict[str, object]] = []
    for spec in MODEL_SNAPSHOT_SPECS:
        model_name, source_run_id = _resolve_model_name(client, spec)
        if model_name not in EXPERIMENT_MODELS:
            raise ValueError(
                f"{spec.registry_name} v{spec.version} resolves to {model_name!r}, "
                f"which is not in EXPERIMENT_MODELS — check the regime table",
            )
        dest = snapshot_model_dir(spec.registry_name, spec.version)
        if dest.is_dir() and not refresh:
            logger.info("Skipping existing %s", dest)
        else:
            if dest.is_dir():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Downloading %s v%d (%s) → %s",
                spec.registry_name,
                spec.version,
                model_name,
                dest,
            )
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{spec.registry_name}/{spec.version}",
            )
            shutil.copytree(local, dest)
        rows.append({
            "registry_name": spec.registry_name,
            "version": spec.version,
            "model_name": model_name,
            "cadence_role": spec.cadence_role,
            "regime": spec.regime,
            "source_run_id": source_run_id,
        })

    manifest = pd.DataFrame(rows, columns=list(MODEL_MANIFEST_COLUMNS))
    _validate_manifest(manifest)
    MODEL_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MODEL_MANIFEST_PATH, index=False)
    logger.info("Wrote %s (%d pins)", MODEL_MANIFEST_PATH, len(manifest))
    return MODEL_MANIFEST_PATH


def _validate_manifest(manifest: pd.DataFrame) -> None:
    """Every (cadence_role, regime) must cover the roster exactly once."""
    expected = set(EXPERIMENT_MODELS)
    for (cadence_role, regime), group in manifest.groupby(["cadence_role", "regime"]):
        got = list(group["model_name"])
        if len(got) != len(set(got)):
            raise ValueError(f"{cadence_role}/{regime}: duplicate model_name in {got}")
        if set(got) != expected:
            raise ValueError(
                f"{cadence_role}/{regime}: pins cover {sorted(got)}, "
                f"expected {sorted(expected)}",
            )
    if manifest["source_run_id"].nunique() != len(manifest):
        raise ValueError("Pinned versions share source runs — the pins are not distinct")


# ---------------------------------------------------------------------------
# Fixture settlement
# ---------------------------------------------------------------------------

_RUNS_DIR: Final[Path] = Path("data/raw/api_football/runs")
# 45 + 15 + 45 minutes is the shortest a match can take, so no snapshot before
# kickoff + 100 min can legitimately show it finished.
_EARLIEST_FINISH: Final[pd.Timedelta] = pd.Timedelta(minutes=100)
# Bronze was committed every 2 h, but the pipeline had outages (the longest gap
# inside the tournament window is 36 h), so probe well past the first candidate.
_PROBE_LIMIT: Final[pd.Timedelta] = pd.Timedelta(days=4)


@dataclass(frozen=True)
class _IngestionRun:
    """One API-Football ingestion, read from its Bronze run manifest."""

    run_id: str
    finished_at: pd.Timestamp
    window_label: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def _load_ingestion_runs(runs_dir: Path = _RUNS_DIR) -> list[_IngestionRun]:
    """Every ingestion run, ascending by completion time."""
    runs: list[_IngestionRun] = []
    for path in sorted(runs_dir.glob("*_run_manifest.json")):
        with path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        window = manifest.get("requested_window") or {}
        finished_at = manifest.get("finished_at_utc")
        if not window.get("label") or not finished_at:
            continue
        runs.append(
            _IngestionRun(
                run_id=manifest["run_id"],
                finished_at=pd.to_datetime(finished_at, utc=True),
                window_label=window["label"],
                start_date=pd.to_datetime(window["start_date"], utc=True),
                end_date=pd.to_datetime(window["end_date"], utc=True),
            )
        )
    runs.sort(key=lambda run: run.finished_at)
    return runs


def _window_covering(
    kickoff: pd.Timestamp,
    commit_time: pd.Timestamp,
    runs: list[_IngestionRun],
) -> str | None:
    """Freshest Bronze window covering ``kickoff`` as of ``commit_time``.

    Windows persist once written, so the newest ingestion that both finished
    before the commit and covered the fixture's date holds the freshest view
    of it in that commit's Bronze.
    """
    day = kickoff.normalize()
    for run in reversed(runs):
        if run.finished_at < commit_time and run.start_date <= day <= run.end_date:
            return run.window_label
    return None


_WINDOW_CACHE: dict[tuple[str, str], Path | None] = {}


def _fetch_window(commit_sha: str, window_label: str) -> Path | None:
    """Materialise one Bronze fixtures window as of ``commit_sha`` (cached)."""
    key = (commit_sha, window_label)
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]

    dest = Path(tempfile.mkdtemp(prefix=f"bronze-{commit_sha[:8]}-")) / window_label
    result = subprocess.run(
        [
            "dvc", "get", ".", f"{RAW_FIXTURES_DVC_PATH}/{window_label}",
            "-o", str(dest), "--rev", commit_sha,
        ],
        capture_output=True,
        text=True,
    )
    path: Path | None = dest / "fixtures.json"
    if result.returncode != 0 or not path.exists():
        logger.warning(
            "No %s in Bronze at %s (%s)",
            window_label,
            commit_sha[:8],
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "missing",
        )
        path = None
    _WINDOW_CACHE[key] = path
    return path


def _status_in_snapshot(path: Path, fixture_id: int) -> str | None:
    """The fixture's status in one Bronze snapshot, or ``None`` if unscored."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for entry in data.get("response", []):
        if entry.get("fixture", {}).get("id") != fixture_id:
            continue
        goals = entry.get("goals", {})
        if goals.get("home") is None or goals.get("away") is None:
            return None
        return entry["fixture"].get("status", {}).get("short", "")
    return None


def _first_known(
    fx: FinishedFixture,
    commits: list[GoldCommit],
    runs: list[_IngestionRun],
) -> tuple[GoldCommit, str, int] | None:
    """First commit whose Bronze shows ``fx`` finished, with its status."""
    probed = 0
    for commit in commits:
        offset = commit.commit_time - fx.kickoff
        if offset < _EARLIEST_FINISH:
            continue
        if offset > _PROBE_LIMIT:
            break
        window_label = _window_covering(fx.kickoff, commit.commit_time, runs)
        if window_label is None:
            continue
        snapshot = _fetch_window(commit.commit_sha, window_label)
        if snapshot is None:
            continue
        probed += 1
        status = _status_in_snapshot(snapshot, fx.fixture_id)
        if status in FINISHED_STATUSES:
            return commit, status, probed
    return None


def build_fixture_settlement_snapshot(*, refresh: bool = False) -> Path:
    """Record, per WC fixture, the first Bronze commit that showed it finished.

    Live inference only knew a result once an ingestion run had written it and
    the pipeline had committed that Bronze alongside the Gold it built from.
    Deriving that from kickoff plus a fixed delay is wrong for matches decided
    in extra time or penalties, or marked finished late, so this walks the
    versioned Bronze instead.  One ``dvc get`` per (commit, window) probed.
    """
    if FIXTURE_SETTLEMENT_PATH.exists() and not refresh:
        logger.info(
            "Skipping existing %s (pass --refresh to rebuild)", FIXTURE_SETTLEMENT_PATH
        )
        return FIXTURE_SETTLEMENT_PATH

    commits = build_gold_commit_index()
    runs = _load_ingestion_runs()
    fixtures = sorted(load_finished_wc_fixtures(), key=lambda fx: fx.kickoff)
    logger.info(
        "Resolving settlement for %d WC fixtures against %d Bronze commits",
        len(fixtures),
        len(commits),
    )

    rows: list[dict[str, object]] = []
    unresolved: list[int] = []
    for fx in fixtures:
        found = _first_known(fx, commits, runs)
        if found is None:
            unresolved.append(fx.fixture_id)
            logger.warning(
                "No Bronze snapshot within %s shows fixture %d (%s vs %s) finished",
                _PROBE_LIMIT,
                fx.fixture_id,
                fx.home_team,
                fx.away_team,
            )
        commit, status, probed = found if found else (None, None, 0)
        rows.append({
            "fixture_id": fx.fixture_id,
            "kickoff_utc": fx.kickoff.isoformat(),
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "round_str": fx.round_str,
            "first_known_utc": commit.commit_time.isoformat() if commit else "",
            "first_known_commit": commit.commit_sha if commit else "",
            "status_at_first_known": status or "",
            "probed_commits": probed,
        })

    settlement = pd.DataFrame(rows, columns=list(FIXTURE_SETTLEMENT_COLUMNS))
    FIXTURE_SETTLEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    settlement.to_csv(FIXTURE_SETTLEMENT_PATH, index=False)
    logger.info(
        "Wrote %s (%d fixtures, %d unresolved)",
        FIXTURE_SETTLEMENT_PATH,
        len(settlement),
        len(unresolved),
    )
    _report_settlement_lag(settlement)
    return FIXTURE_SETTLEMENT_PATH


def _report_settlement_lag(settlement: pd.DataFrame) -> None:
    """Log how long after kickoff each result became visible."""
    resolved = settlement[settlement["first_known_utc"] != ""].copy()
    if resolved.empty:
        return
    lag = (
        pd.to_datetime(resolved["first_known_utc"], utc=True, format="mixed")
        - pd.to_datetime(resolved["kickoff_utc"], utc=True, format="mixed")
    ).dt.total_seconds() / 3600
    logger.info(
        "Visibility lag after kickoff: min %.2f h, median %.2f h, max %.2f h",
        lag.min(),
        lag.median(),
        lag.max(),
    )
    late = resolved.loc[lag > 2.6, ["home_team", "away_team", "status_at_first_known"]]
    for row, hours in zip(late.itertuples(), lag[lag > 2.6], strict=False):
        logger.info(
            "  late: %s vs %s (%s) visible %.2f h after kickoff",
            row.home_team,
            row.away_team,
            row.status_at_first_known,
            hours,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild outputs that already exist",
    )
    parser.add_argument(
        "--only",
        choices=("predictions", "models", "settlement"),
        default=None,
        help="Snapshot only one part (default: all three)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.only in (None, "models"):
        build_model_snapshot(refresh=args.refresh)
    if args.only in (None, "predictions"):
        build_prediction_snapshot(refresh=args.refresh)
    if args.only in (None, "settlement"):
        build_fixture_settlement_snapshot(refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
