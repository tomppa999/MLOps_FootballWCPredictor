import pandas as pd
import pytest

from src.analysis.entropy_refit_boundary import boundary_changes
from src.analysis.rq2_paired import STAGES


def _frame(first: float) -> pd.DataFrame:
    return pd.DataFrame([
        {"inference_run_id": run, "inference_timestamp": timestamp,
         "model_name": "poisson_glm", "cadence_mode": "per_round",
         **dict.fromkeys(STAGES, value)}
        for run, timestamp, value in [
            ("before", "2026-06-18T13:00:00Z", first),
            ("after", "2026-06-18T14:09:00Z", 3.0),
        ]
    ])


def test_keeps_plotted_and_live_baselines_distinct_for_every_target() -> None:
    result = boundary_changes(_frame(2.0), _frame(2.5), "after")
    assert len(result) == 12
    assert list(result.target.unique()) == [s.removeprefix("entropy_") for s in STAGES]
    assert result[result.baseline == "live_per_round"].increase_percent.tolist() == [50.0] * 6
    assert result[result.baseline == "plotted"].increase_percent.tolist() == [20.0] * 6


def test_rejects_missing_refit_run() -> None:
    with pytest.raises(ValueError, match="post-refit"):
        boundary_changes(_frame(2.0), _frame(2.5), "missing")


def test_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="baseline"):
        boundary_changes(_frame(0.0), _frame(2.5), "after")
