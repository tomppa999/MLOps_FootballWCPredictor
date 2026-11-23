"""Summarise all six Poisson entropy targets at the first valid MD1 refit."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.plot_entropy_trajectory import load_entropy
from src.analysis.rq2_paired import STAGES
from src.analysis.rq_datasets.paths import RQ1_PATH, RQ2_PATH

OUTPUT = Path("docs/tables/rq2_refit_boundary.csv")


def boundary_changes(raw: pd.DataFrame, plotted: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Compare the post-refit snapshot with both explicitly labelled baselines."""
    frames = {}
    for name, frame in (("live_per_round", raw), ("plotted", plotted)):
        frame = frame[
            (frame.model_name == "poisson_glm") & (frame.cadence_mode == "per_round")
        ].copy()
        frame["inference_timestamp"] = pd.to_datetime(
            frame.inference_timestamp, utc=True, format="mixed",
        )
        frames[name] = frame.sort_values("inference_timestamp")
    post = frames["live_per_round"].loc[lambda frame: frame.inference_run_id == run_id]
    if len(post) != 1:
        raise ValueError("Expected exactly one Poisson post-refit snapshot")
    after = post.iloc[0]
    rows = []
    for baseline, frame in frames.items():
        prior = frame[frame.inference_timestamp < after.inference_timestamp]
        if prior.empty:
            raise ValueError(f"Missing pre-refit snapshot for {baseline}")
        before = prior.iloc[-1]
        for stage in STAGES:
            first, last = float(before[stage]), float(after[stage])
            if not np.isfinite([first, last]).all() or first <= 0:
                raise ValueError(f"Invalid entropy baseline for {stage}")
            rows.append({
                "baseline": baseline, "target": stage.removeprefix("entropy_"),
                "before_timestamp": before.inference_timestamp,
                "after_timestamp": after.inference_timestamp,
                "before_run_id": before.inference_run_id, "after_run_id": run_id,
                "before_nats": first, "after_nats": last, "increase_nats": last - first,
                "increase_percent": 100 * (last - first) / first,
            })
    return pd.DataFrame(rows)


def main() -> None:
    matches = pd.read_csv(RQ1_PATH, dtype={"round_label": str})
    md2 = matches[
        (matches.round_label == "2") & (matches.model_name == "xgboost")
        & (matches.cadence_mode == "per_round")
    ].copy()
    md2["kickoff_utc"] = pd.to_datetime(md2.kickoff_utc, utc=True, format="mixed")
    run_id = md2.sort_values("kickoff_utc").iloc[0].inference_run_id
    changes = boundary_changes(pd.read_csv(RQ2_PATH), load_entropy(), run_id)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    changes.to_csv(OUTPUT, index=False)
    print(changes[["baseline", "target", "before_nats", "after_nats", "increase_nats", "increase_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
