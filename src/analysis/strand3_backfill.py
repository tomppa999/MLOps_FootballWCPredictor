"""Strand 3: backfill four dropped never-refit shadow rows.

All four rows belong to models that are never refitted, so both cadences must
carry identical predictions.  Each row is therefore copied byte-for-byte from
its twin in the other cadence rather than re-predicted offline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.replay_common import (
    BACKFILL_FIXTURES,
    ensure_output_dir,
    load_monitoring_artifact,
    log_reconstruction_run,
)
from src.analysis.rq_datasets.export_live import monitoring_path

logger = logging.getLogger(__name__)


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

    source_paths = {
        "per_round": per_round_monitoring_path or monitoring_path("per_round"),
        "frozen": frozen_monitoring_path or monitoring_path("frozen"),
    }
    sources: dict[str, pd.DataFrame] = {}
    for cadence, path in source_paths.items():
        if Path(path).exists():
            sources[cadence] = load_monitoring_artifact(Path(path))
        else:
            logger.warning("No %s monitoring artifact at %s", cadence, path)

    backfill_rows: list[dict] = []
    for label, spec in BACKFILL_FIXTURES.items():
        source_cadence = spec["copy_from_cadence"]
        if source_cadence not in sources:
            raise FileNotFoundError(
                f"{label}: needs {source_cadence} monitoring at "
                f"{source_paths[source_cadence]}",
            )
        for fixture_id in spec["fixture_ids"]:
            backfill_rows.append(
                _copy_row_from_monitoring(
                    sources[source_cadence],
                    fixture_id,
                    spec["model_name"],
                    source_cadence,
                    spec["cadence_mode"],
                ),
            )
        logger.info(
            "%s: copied %d row(s) from %s",
            label,
            len(spec["fixture_ids"]),
            source_cadence,
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
