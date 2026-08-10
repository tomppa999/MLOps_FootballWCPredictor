"""Strand 3: backfill four dropped never-refit shadow rows."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.replay_common import (
    BACKFILL_FIXTURES,
    augment_gold_for_inference,
    build_gold_commit_index,
    ensure_output_dir,
    load_gold_at_commit,
    load_pinned_shadow_model,
    load_monitoring_artifact,
    log_reconstruction_run,
    match_row_for_fixture,
    parse_wc_results_before_kickoff,
    predict_single_model,
    resolve_gold_commit,
    score_prediction_row,
)
from src.models.mlflow_utils import (
    SHADOW_MODEL_NAME,
    _latest_version_with_tags,
    setup_mlflow,
)
from src.monitoring.monitor import parse_wc_settled_matches

logger = logging.getLogger(__name__)


def _resolve_never_refit_version(model_name: str) -> int:
    """Return the wc_shadow version for a never-refit model (cadence-invariant)."""
    setup_mlflow()
    mv = _latest_version_with_tags(SHADOW_MODEL_NAME, model_name)
    if mv is None:
        raise ValueError(f"No wc_shadow version for {model_name}")
    return int(mv.version)


def _copy_row_from_monitoring(
    monitoring_df: pd.DataFrame,
    fixture_id: int,
    model_name: str,
    source_cadence: str,
    target_cadence: str,
) -> dict:
    row = monitoring_df[
        (monitoring_df["match_id"] == fixture_id)
        & (monitoring_df["model_name"] == model_name)
        & (monitoring_df["cadence_mode"] == source_cadence)
    ]
    if row.empty:
        raise KeyError(
            f"No {source_cadence} row for {model_name} fixture {fixture_id}",
        )
    out = row.iloc[0].to_dict()
    out["cadence_mode"] = target_cadence
    out["inference_run_id"] = f"backfill_copy_from_{source_cadence}"
    return out


def run_strand3_backfill(
    *,
    per_round_monitoring_path: Path | None = None,
    frozen_monitoring_path: Path | None = None,
) -> str:
    """Backfill the four dropped model-match rows to 832/832 per cadence."""
    out_dir = ensure_output_dir("strand3_backfill")
    settled = parse_wc_settled_matches()
    gold_index = build_gold_commit_index()
    backfill_rows: list[dict] = []

    per_round_mon = (
        load_monitoring_artifact(per_round_monitoring_path)
        if per_round_monitoring_path
        else pd.DataFrame()
    )
    frozen_mon = (
        load_monitoring_artifact(frozen_monitoring_path)
        if frozen_monitoring_path
        else pd.DataFrame()
    )

    # Two MD3 per_round ridge rows — re-predict.
    ridge_version = _resolve_never_refit_version("ridge")
    ridge_model = load_pinned_shadow_model("ridge", ridge_version)
    for fixture_id in BACKFILL_FIXTURES["ridge_per_round_md3"]["fixture_ids"]:
        match = match_row_for_fixture(settled, fixture_id)
        commit = resolve_gold_commit(match["kickoff_utc"], gold_index)
        if commit is None:
            raise RuntimeError(f"No Gold commit for fixture {fixture_id}")
        gold_df = load_gold_at_commit(commit.commit_sha)
        wc_partial = parse_wc_results_before_kickoff(match["kickoff_utc"])
        augmented, ref_date = augment_gold_for_inference(gold_df, wc_partial)
        pred = predict_single_model(
            ridge_model,
            "ridge",
            match["home"],
            match["away"],
            augmented,
            ref_date,
        )
        backfill_rows.append(
            score_prediction_row(
                match,
                "ridge",
                pred,
                cadence_mode="per_round",
                inference_run_id="backfill_ridge_md3",
            ),
        )

    # random_forest frozen — copy from per_round twin.
    if not per_round_mon.empty:
        for fixture_id in BACKFILL_FIXTURES["random_forest_frozen_r32"]["fixture_ids"]:
            backfill_rows.append(
                _copy_row_from_monitoring(
                    per_round_mon,
                    fixture_id,
                    "random_forest",
                    "per_round",
                    "frozen",
                ),
            )
        for fixture_id in BACKFILL_FIXTURES["mean_rate_poisson_frozen_md1"]["fixture_ids"]:
            backfill_rows.append(
                _copy_row_from_monitoring(
                    per_round_mon,
                    fixture_id,
                    "mean_rate_poisson",
                    "per_round",
                    "frozen",
                ),
            )

    df = pd.DataFrame(backfill_rows)
    csv_path = out_dir / "backfill_rows.csv"
    df.to_csv(csv_path, index=False)

    return log_reconstruction_run(
        strand="strand3_backfill",
        params={"backfill_row_count": str(len(df))},
        metrics={"rows_backfilled": float(len(df))},
        artifacts={"backfill_rows": csv_path},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_strand3_backfill()
