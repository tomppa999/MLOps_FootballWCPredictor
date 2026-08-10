"""Strand 1: offline frozen-shadow reconstruction for poisson_glm and bayesian_poisson."""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from src.analysis.replay_common import (
    PINNED_FROZEN_SHADOW_VERSIONS,
    augment_gold_for_inference,
    build_gold_commit_index,
    ensure_output_dir,
    leaderboard_summary,
    load_gold_at_commit,
    load_pinned_shadow_model,
    log_reconstruction_run,
    parse_wc_results_before_kickoff,
    predict_single_model,
    resolve_gold_commit,
    score_prediction_row,
)
from src.monitoring.monitor import parse_wc_settled_matches

logger = logging.getLogger(__name__)


def run_strand1_frozen_shadow() -> dict[str, str]:
    """Rebuild true-frozen monitoring rows for pinned shadow models."""
    out_dir = ensure_output_dir("strand1_frozen_shadow")
    settled = parse_wc_settled_matches()
    gold_index = build_gold_commit_index()

    all_rows: list[dict] = []
    run_ids: dict[str, str] = {}

    for model_name, version in PINNED_FROZEN_SHADOW_VERSIONS.items():
        logger.info("Loading pinned frozen shadow %s v%s", model_name, version)
        model = load_pinned_shadow_model(model_name, version)
        by_commit: dict[str, list[pd.Series]] = defaultdict(list)

        for _, match in settled.iterrows():
            commit = resolve_gold_commit(match["kickoff_utc"], gold_index)
            if commit is None:
                continue
            by_commit[commit.commit_sha].append(match)

        model_rows: list[dict] = []
        for commit_sha, matches in by_commit.items():
            gold_df = load_gold_at_commit(commit_sha)
            for match in matches:
                wc_partial = parse_wc_results_before_kickoff(match["kickoff_utc"])
                augmented, ref_date = augment_gold_for_inference(gold_df, wc_partial)
                pred = predict_single_model(
                    model,
                    model_name,
                    match["home"],
                    match["away"],
                    augmented,
                    ref_date,
                )
                model_rows.append(
                    score_prediction_row(
                        match,
                        model_name,
                        pred,
                        cadence_mode="frozen",
                    ),
                )

        df = pd.DataFrame(model_rows)
        csv_path = out_dir / f"frozen_shadow_{model_name}.csv"
        df.to_csv(csv_path, index=False)
        summary = leaderboard_summary(df)
        summary_path = out_dir / f"frozen_shadow_{model_name}_summary.csv"
        summary.to_csv(summary_path, index=False)
        all_rows.extend(model_rows)

        run_id = log_reconstruction_run(
            strand="strand1_frozen_shadow",
            params={
                "model_name": model_name,
                "shadow_version": str(version),
                "cadence_mode": "frozen",
                "match_count": str(len(df)),
            },
            metrics={
                "mean_rps": float(df["rps"].mean()) if not df.empty else float("nan"),
                "mean_nll": float(df["nll"].mean()) if not df.empty else float("nan"),
                "mean_rmse": float(
                    (df["rmse_h"].mean() + df["rmse_a"].mean()) / 2,
                ) if not df.empty else float("nan"),
            },
            artifacts={
                "rows": csv_path,
                "summary": summary_path,
            },
            tags={"model_name": model_name},
        )
        run_ids[model_name] = run_id
        logger.info(
            "%s frozen rebuild: %d rows, mean_rps=%.4f, run_id=%s",
            model_name,
            len(df),
            df["rps"].mean() if not df.empty else float("nan"),
            run_id,
        )

    combined = pd.DataFrame(all_rows)
    combined_path = out_dir / "frozen_shadow_combined.csv"
    combined.to_csv(combined_path, index=False)
    combined_summary = leaderboard_summary(combined)
    combined_summary_path = out_dir / "frozen_shadow_leaderboard.csv"
    combined_summary.to_csv(combined_summary_path, index=False)

    run_ids["combined"] = log_reconstruction_run(
        strand="strand1_frozen_shadow",
        params={"scope": "combined_leaderboard"},
        artifacts={
            "combined": combined_path,
            "leaderboard": combined_summary_path,
        },
    )
    return run_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_strand1_frozen_shadow()
