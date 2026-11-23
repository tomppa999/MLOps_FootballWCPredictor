"""Checks for the isolated Poisson refit investigation."""

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from src.analysis.poisson_refit_diagnostic import (
    fit_with_start,
    numeric_state,
    rebase_params,
)


def _state() -> dict[str, np.ndarray]:
    return {
        "mean_": np.array([1.0, 2.0]),
        "scale_": np.array([2.0, 3.0]),
        "_coef1": np.array([0.1, 0.2, 0.3]),
        "_coef2": np.array([-0.1, -0.2, -0.3]),
        "_coef3": np.array([-2.0, 0.4, -0.5]),
    }


def test_numeric_state_reads_arrays_without_unpickling(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(_state(), protocol=5))
    with patch("pickle.loads", side_effect=AssertionError("must not deserialize")):
        actual = numeric_state(path, n_features=2)
    for key, expected in _state().items():
        np.testing.assert_array_equal(actual[key], expected)


def test_numeric_state_rejects_missing_or_invalid_arrays(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    state = _state()
    state["scale_"][0] = 0
    path.write_bytes(pickle.dumps(state, protocol=5))
    with pytest.raises(ValueError, match="scale"):
        numeric_state(path, n_features=2)
    path.write_bytes(pickle.dumps({"mean_": np.zeros(2)}, protocol=5))
    with pytest.raises(ValueError, match="Missing"):
        numeric_state(path, n_features=2)


def test_rebase_preserves_all_three_component_predictions() -> None:
    state = _state()
    mean, scale = np.array([4.0, -1.0]), np.array([0.5, 7.0])
    x = np.array([[0.0, 1.0], [5.0, -3.0]])
    params = rebase_params(state, mean, scale).reshape(3, 3)
    old_design = np.column_stack([np.ones(len(x)), (x - state["mean_"]) / state["scale_"]])
    new_design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    for i in range(3):
        np.testing.assert_allclose(
            new_design @ params[i], old_design @ state[f"_coef{i + 1}"], atol=1e-14,
        )


def test_failed_diagnostic_fit_keeps_iterate_and_reports_failure() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.ones((3, 2))
    final = np.arange(6, dtype=float) / 10
    result = OptimizeResult(x=final, success=False, message="iteration limit")
    with patch("src.analysis.poisson_refit_diagnostic.minimize", return_value=result) as run:
        model, actual = fit_with_start(x, y, np.ones(3), start=None, alpha=1.0)
    assert not actual.success
    np.testing.assert_array_equal(model._coef1, final[:2])
    np.testing.assert_array_equal(run.call_args.args[1], np.zeros(6))
    assert run.call_args.kwargs["options"] == {"maxiter": 500, "ftol": 1e-9}


def test_strict_refinement_increases_solver_budget() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    result = OptimizeResult(x=np.zeros(6), success=True)
    with patch("src.analysis.poisson_refit_diagnostic.minimize", return_value=result) as run:
        fit_with_start(x, np.ones((3, 2)), np.ones(3), start=None, alpha=1.0, strict=True)
    assert run.call_args.kwargs["options"] == {
        "maxiter": 500, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 100000,
    }
