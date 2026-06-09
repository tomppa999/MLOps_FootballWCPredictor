"""Export scoreline distributions as per-stage CSV files for betting analysis.

Usage (from project root):

    python -m src.inference.export                      # latest frozen run
    python -m src.inference.export --run-id <run_id>    # specific run
    python -m src.inference.export --top-n 5            # top-5 per match (default: 3)
    python -m src.inference.export --out-dir /tmp/bets  # custom output dir
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_STAGE_ORDER = {"Group": 0, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "Final": 5}


def export_scoreline_csvs(
    output_dir: Path | str = Path("exports/scorelines"),
    run_id: str | None = None,
    top_n: int = 3,
) -> dict[str, Path]:
    """Export top-N scorelines per match, split by stage, as CSV files.

    Args:
        output_dir: Directory to write CSVs into (created if absent).
        run_id: Specific MLflow run ID to load from. When ``None`` the latest
            frozen inference run is used (same run the dashboard shows).
        top_n: How many scorelines to keep per match (sorted by probability).

    Returns:
        Mapping of stage name -> written CSV path.
    """
    from src.dashboard.load_artifacts import (
        _get_latest_inference_run,
        load_latest_inference_artifacts,
    )
    import mlflow

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        data, info = load_latest_inference_artifacts()
        used_run_id = info.run_id
        sl_df = data.get("scoreline_distributions")
    else:
        setup_fn = None
        try:
            from src.models.mlflow_utils import setup_mlflow
            setup_fn = setup_mlflow
        except ImportError:
            pass
        if setup_fn:
            setup_fn()
        client = mlflow.tracking.MlflowClient()
        artifact_dir = Path(client.download_artifacts(run_id, ""))
        path = artifact_dir / "scoreline_distributions.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"scoreline_distributions.csv not found in run {run_id}. "
                "Re-run inference to generate it."
            )
        sl_df = pd.read_csv(path)
        used_run_id = run_id

    if sl_df is None or sl_df.empty:
        raise RuntimeError(
            "scoreline_distributions artifact is empty or missing. "
            "Re-run inference to regenerate it."
        )

    required = {"home_team", "away_team", "home_goals", "away_goals", "probability"}
    missing = required - set(sl_df.columns)
    if missing:
        raise ValueError(
            f"scoreline_distributions is missing columns: {missing}. "
            "Re-run inference to regenerate the artifact."
        )

    # Add stage column if absent (legacy runs before stage was logged).
    if "stage" not in sl_df.columns:
        sl_df["stage"] = "Group"

    written: dict[str, Path] = {}
    stages = sorted(sl_df["stage"].dropna().unique(), key=lambda s: _STAGE_ORDER.get(s, 99))

    for stage in stages:
        stage_df = sl_df[sl_df["stage"] == stage].copy()

        rows = []
        for (home, away), match_sl in stage_df.groupby(["home_team", "away_team"]):
            top = (
                match_sl.nlargest(top_n, "probability")
                .reset_index(drop=True)
            )
            for rank, r in top.iterrows():
                rows.append({
                    "stage": stage,
                    "home_team": home,
                    "away_team": away,
                    "rank": int(rank) + 1,
                    "home_goals": int(r["home_goals"]),
                    "away_goals": int(r["away_goals"]),
                    "scoreline": f"{int(r['home_goals'])}-{int(r['away_goals'])}",
                    "probability": round(float(r["probability"]), 4),
                    "probability_pct": round(float(r["probability"]) * 100, 2),
                })

        if not rows:
            continue

        out_df = pd.DataFrame(rows)
        filename = f"scorelines_{stage.lower()}.csv"
        path = out / filename
        out_df.to_csv(path, index=False)
        written[stage] = path
        logger.info("Wrote %d rows → %s", len(out_df), path)

    logger.info("Export complete. run_id=%s, stages=%s", used_run_id, list(written.keys()))
    return written


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Export scoreline CSVs per stage.")
    parser.add_argument("--run-id", default=None, help="MLflow run ID (default: latest frozen)")
    parser.add_argument("--top-n", type=int, default=3, help="Top N scorelines per match")
    parser.add_argument(
        "--out-dir",
        default="exports/scorelines",
        help="Output directory (default: exports/scorelines)",
    )
    args = parser.parse_args()
    written = export_scoreline_csvs(
        output_dir=args.out_dir,
        run_id=args.run_id,
        top_n=args.top_n,
    )
    for stage, path in written.items():
        print(f"  {stage:10s}  →  {path}")


if __name__ == "__main__":
    _main()
