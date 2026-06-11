"""Mirror the MLflow registry from DagsHub into a local file store.

Backup direction is DagsHub (primary) -> local (`file:./mlruns`). Run it after
each pipeline cycle so a DagsHub outage never blocks offline serving: the trained
model weights (incl. the non-reproducible bayesian_poisson MCMC posterior) are
copied verbatim, registered locally, and the production aliases are reproduced.

After a successful mirror you can serve fully offline::

    export MLFLOW_TRACKING_URI=file:./mlruns
    python -m src.pipeline.trigger --mode=inference_only   # or run the dashboard

What it copies (idempotent — only adds a local version when the source changed):
  * ``wc_production`` versions behind the ``champion_frozen`` /
    ``champion_per_round`` / ``champion`` aliases, with the aliases re-applied.
  * ``wc_shadow`` versions for the roster models (the cadence-tagged + fallback
    versions that ``load_shadow_model`` resolves), so both cadence modes work
    offline. Falls back to ``wc_staging`` when a model has no shadow version yet.

Scheduling (decoupled from GCP — runs where the durable disk lives, e.g. your
Mac). Example launchd / cron entry, ~15 min after the GCP cycle::

    # crontab -e  (runs hourly at :20, logs to a file)
    20 * * * * cd /path/to/repo && /path/to/conda/envs/modelops/bin/python \
        src/backup/mirror_from_dagshub.py >> logs/mirror.log 2>&1

Usage::

    python src/backup/mirror_from_dagshub.py [--target file:./mlruns] [--all-shadows]
        [--skip-production] [--skip-shadows] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Bound MLflow's HTTP client so a DagsHub outage fails fast instead of hanging
# on multi-minute exponential backoff. Must be set before mlflow is imported.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "120")

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

# Allow running as a plain file (e.g. from cron): put the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importing mlflow_utils runs load_dotenv(), so MLFLOW_TRACKING_URI (DagsHub)
# and its credentials are populated from .env before we read them below.
from src.models.config import EXPERIMENT_MODELS, LIVE_SHADOW_MODELS
from src.models.mlflow_utils import (
    CHAMPION_ALIAS_DEFAULT,
    CHAMPION_ALIAS_FROZEN,
    CHAMPION_ALIAS_PER_ROUND,
    PRODUCTION_MODEL_NAME,
    SHADOW_MODEL_NAME,
    STAGING_MODEL_NAME,
)

logger = logging.getLogger("mirror_from_dagshub")

DEFAULT_TARGET = "file:./mlruns"
MIRROR_EXPERIMENT = "registry_mirror"
CADENCE_MODES = ("frozen", "per_round")
PRODUCTION_ALIASES = (
    CHAMPION_ALIAS_FROZEN,
    CHAMPION_ALIAS_PER_ROUND,
    CHAMPION_ALIAS_DEFAULT,
)
# MLflow-reserved tags can't be re-set on a new run; copy everything else.
_RESERVED_TAG_PREFIX = "mlflow."


@dataclass
class MirrorItem:
    """A single source registry version queued for mirroring."""

    registered_model: str
    source_version: str
    source_run_id: str
    source_uri: str
    params: dict[str, str]
    tags: dict[str, str]
    metrics: dict[str, float]
    aliases: list[str] = field(default_factory=list)
    staged_dir: Path | None = None  # populated after download; None if skipped
    already_present_version: str | None = None  # local version if idempotent-skip


def _run_data(client: MlflowClient, run_id: str) -> tuple[dict, dict, dict]:
    """Return (params, tags, metrics) for a run, or empty dicts on failure."""
    try:
        data = client.get_run(run_id).data
    except Exception as exc:  # noqa: BLE001 — network/registry errors must not abort the batch
        logger.warning("Could not read run %s: %s", run_id, exc)
        return {}, {}, {}
    return dict(data.params), dict(data.tags), dict(data.metrics)


def _existing_local_version(
    tgt_client: MlflowClient, registered_model: str, source_run_id: str,
) -> str | None:
    """Return the local version already mirrored from ``source_run_id``, if any."""
    try:
        versions = tgt_client.search_model_versions(f"name='{registered_model}'")
    except Exception:  # noqa: BLE001
        return None
    for mv in versions:
        try:
            tags = tgt_client.get_run(mv.run_id).data.tags
        except Exception:  # noqa: BLE001
            continue
        if tags.get("mirror_source_run_id") == source_run_id:
            return mv.version
    return None


def _latest_version_with_tags(
    client: MlflowClient,
    registered_model: str,
    model_name: str,
    cadence_mode: str | None,
) -> ModelVersion | None:
    """Source-side replica of mlflow_utils._latest_version_with_tags.

    Newest version whose run has the matching ``model_name`` tag (and, when
    given, ``cadence_mode``). Mirrors the resolution order used at inference.
    """
    try:
        versions = client.search_model_versions(f"name='{registered_model}'")
    except Exception:  # noqa: BLE001
        return None
    versions = sorted(versions, key=lambda mv: int(mv.version), reverse=True)

    def tags_of(mv: ModelVersion) -> dict | None:
        try:
            return client.get_run(mv.run_id).data.tags
        except Exception:  # noqa: BLE001
            return None

    if cadence_mode is not None:
        for mv in versions:
            t = tags_of(mv)
            if t and t.get("model_name") == model_name and t.get("cadence_mode") == cadence_mode:
                return mv
    for mv in versions:
        t = tags_of(mv)
        if t and t.get("model_name") == model_name:
            return mv
    return None


def _collect_production_items(src_client: MlflowClient) -> list[MirrorItem]:
    """Resolve the production alias versions into mirror items (deduped)."""
    by_version: dict[str, MirrorItem] = {}
    for alias in PRODUCTION_ALIASES:
        try:
            mv = src_client.get_model_version_by_alias(PRODUCTION_MODEL_NAME, alias)
        except Exception as exc:  # noqa: BLE001
            logger.info("Alias %r unavailable on %s (%s) — skipping.", alias, PRODUCTION_MODEL_NAME, exc)
            continue
        item = by_version.get(mv.version)
        if item is None:
            params, tags, metrics = _run_data(src_client, mv.run_id)
            item = MirrorItem(
                registered_model=PRODUCTION_MODEL_NAME,
                source_version=mv.version,
                source_run_id=mv.run_id,
                source_uri=mv.source,
                params=params,
                tags=tags,
                metrics=metrics,
            )
            by_version[mv.version] = item
        item.aliases.append(alias)
        logger.info("Production: alias %r -> v%s (run %s)", alias, mv.version, mv.run_id)
    return list(by_version.values())


def _collect_shadow_items(
    src_client: MlflowClient, model_names: list[str],
) -> list[MirrorItem]:
    """Resolve the shadow versions inference would load for each roster model."""
    by_key: dict[tuple[str, str], MirrorItem] = {}
    for name in model_names:
        picked: dict[str, ModelVersion] = {}
        for mode in (*CADENCE_MODES, None):
            mv = _latest_version_with_tags(src_client, SHADOW_MODEL_NAME, name, mode)
            registered = SHADOW_MODEL_NAME
            if mv is None:
                mv = _latest_version_with_tags(src_client, STAGING_MODEL_NAME, name, mode)
                registered = STAGING_MODEL_NAME
            if mv is None:
                continue
            picked[mv.version] = mv
            key = (registered, mv.version)
            if key in by_key:
                continue
            params, tags, metrics = _run_data(src_client, mv.run_id)
            by_key[key] = MirrorItem(
                registered_model=registered,
                source_version=mv.version,
                source_run_id=mv.run_id,
                source_uri=mv.source,
                params=params,
                tags=tags,
                metrics=metrics,
            )
        if not picked:
            logger.warning("No shadow/staging version found for model_name=%r — skipping.", name)
        else:
            logger.info("Shadow %r -> versions %s", name, sorted(picked))
    return list(by_key.values())


def _download(item: MirrorItem, source_uri: str, staging_root: Path) -> bool:
    """Download the item's model artifacts into the staging dir. Returns success."""
    dst = staging_root / f"{item.registered_model}_v{item.source_version}"
    dst.mkdir(parents=True, exist_ok=True)
    try:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=item.source_uri,
            dst_path=str(dst),
            tracking_uri=source_uri,
        )
    except Exception as exc:  # noqa: BLE001 — network/registry errors must not abort the batch
        logger.error(
            "Download failed for %s v%s (%s): %s",
            item.registered_model, item.source_version, item.source_uri, exc,
        )
        return False
    item.staged_dir = Path(local)
    return True


def _clean_tags(tags: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in tags.items() if not k.startswith(_RESERVED_TAG_PREFIX)}


def _clean_metrics(metrics: dict[str, float]) -> dict[str, float]:
    import math

    return {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}


def _relog_and_register(
    item: MirrorItem, tgt_client: MlflowClient, experiment_id: str,
) -> str:
    """Re-log the staged model to the target store and register it. Returns version."""
    extra = {
        "mirror_source_run_id": item.source_run_id,
        "mirror_source_version": item.source_version,
        "mirror_source_model": item.registered_model,
        "mirror_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"mirror_{item.registered_model}_v{item.source_version}",
    ) as run:
        mlflow.set_tags({**_clean_tags(item.tags), **extra})
        if item.params:
            mlflow.log_params(item.params)
        clean_metrics = _clean_metrics(item.metrics)
        if clean_metrics:
            mlflow.log_metrics(clean_metrics)
        assert item.staged_dir is not None
        mlflow.log_artifacts(str(item.staged_dir), artifact_path="model")
        local_run_id = run.info.run_id

    mv = mlflow.register_model(f"runs:/{local_run_id}/model", item.registered_model)
    return mv.version


def _apply_aliases(
    tgt_client: MlflowClient, registered_model: str, version: str, aliases: list[str],
) -> None:
    for alias in aliases:
        tgt_client.set_registered_model_alias(registered_model, alias, version)
        logger.info("Local alias %s@%s -> v%s", registered_model, alias, version)


def mirror(
    *,
    source_uri: str,
    target_uri: str,
    do_production: bool,
    do_shadows: bool,
    shadow_models: list[str],
    dry_run: bool,
) -> dict:
    """Run the mirror. Returns a manifest dict (also written to disk)."""
    src_client = MlflowClient(tracking_uri=source_uri)
    tgt_client = MlflowClient(tracking_uri=target_uri)

    logger.info("Source: %s", source_uri)
    logger.info("Target: %s", target_uri)

    items: list[MirrorItem] = []
    if do_production:
        items += _collect_production_items(src_client)
    if do_shadows:
        items += _collect_shadow_items(src_client, shadow_models)

    if not items:
        logger.warning("No source versions resolved — nothing to mirror.")

    # Idempotency: skip items already mirrored (same source run_id) but still
    # ensure their aliases point at the existing local version.
    to_download: list[MirrorItem] = []
    for item in items:
        existing = _existing_local_version(tgt_client, item.registered_model, item.source_run_id)
        if existing is not None:
            item.already_present_version = existing
            logger.info(
                "Already mirrored: %s (source run %s) -> local v%s",
                item.registered_model, item.source_run_id, existing,
            )
        else:
            to_download.append(item)

    manifest = {
        "source": source_uri,
        "target": target_uri,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mirrored": [],
        "skipped_existing": [],
        "failed": [],
    }

    if dry_run:
        for item in items:
            status = "skip-existing" if item.already_present_version else "would-mirror"
            logger.info(
                "[dry-run] %s %s v%s aliases=%s",
                status, item.registered_model, item.source_version, item.aliases,
            )
        return manifest

    with tempfile.TemporaryDirectory(prefix="mlflow_mirror_") as tmp:
        staging_root = Path(tmp)
        downloaded: list[MirrorItem] = []
        for item in to_download:
            if _download(item, source_uri, staging_root):
                downloaded.append(item)
            else:
                manifest["failed"].append(
                    {"model": item.registered_model, "version": item.source_version}
                )

        # Switch the active store to the target for all writes.
        mlflow.set_tracking_uri(target_uri)
        exp = tgt_client.get_experiment_by_name(MIRROR_EXPERIMENT)
        experiment_id = exp.experiment_id if exp else tgt_client.create_experiment(MIRROR_EXPERIMENT)

        for item in downloaded:
            try:
                local_version = _relog_and_register(item, tgt_client, experiment_id)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Re-log/register failed for %s v%s: %s",
                    item.registered_model, item.source_version, exc,
                )
                manifest["failed"].append(
                    {"model": item.registered_model, "version": item.source_version}
                )
                continue
            _apply_aliases(tgt_client, item.registered_model, local_version, item.aliases)
            manifest["mirrored"].append({
                "model": item.registered_model,
                "source_version": item.source_version,
                "local_version": local_version,
                "source_run_id": item.source_run_id,
                "aliases": item.aliases,
            })

    # Re-apply aliases for idempotent-skipped items (alias may have moved).
    for item in items:
        if item.already_present_version and item.aliases:
            _apply_aliases(
                tgt_client, item.registered_model, item.already_present_version, item.aliases,
            )
            manifest["skipped_existing"].append({
                "model": item.registered_model,
                "local_version": item.already_present_version,
                "aliases": item.aliases,
            })

    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="Source tracking URI (default: $MLFLOW_TRACKING_URI from .env = DagsHub).",
    )
    p.add_argument("--target", default=DEFAULT_TARGET, help=f"Target store (default: {DEFAULT_TARGET}).")
    p.add_argument("--skip-production", action="store_true", help="Do not mirror wc_production.")
    p.add_argument("--skip-shadows", action="store_true", help="Do not mirror shadow models.")
    p.add_argument(
        "--all-shadows",
        action="store_true",
        help="Mirror all LIVE_SHADOW_MODELS (default: the EXPERIMENT_MODELS roster only).",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve and report; do not download/write.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # urllib3 logs every retry at WARNING; bounded retries already fail fast and
    # this script reports its own failures, so keep its output quiet.
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    args = _parse_args(argv)

    if not args.source:
        logger.error("No source tracking URI. Set MLFLOW_TRACKING_URI or pass --source.")
        return 2

    shadow_models = list(LIVE_SHADOW_MODELS if args.all_shadows else EXPERIMENT_MODELS)

    manifest = mirror(
        source_uri=args.source,
        target_uri=args.target,
        do_production=not args.skip_production,
        do_shadows=not args.skip_shadows,
        shadow_models=shadow_models,
        dry_run=args.dry_run,
    )

    manifest_path = Path("logs") / "mirror_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    n_ok = len(manifest["mirrored"])
    n_skip = len(manifest["skipped_existing"])
    n_fail = len(manifest["failed"])
    logger.info(
        "Done. mirrored=%d skipped_existing=%d failed=%d (manifest: %s)",
        n_ok, n_skip, n_fail, manifest_path,
    )
    return 1 if n_fail and not (n_ok or n_skip) else 0


if __name__ == "__main__":
    raise SystemExit(main())
