"""Approved core Holm families; synthetic summaries, no production output writes."""

from __future__ import annotations

import csv
from importlib import import_module
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.rq1_paired import holm_core_families

MODELS = ("xgboost", "poisson_glm", "bayesian_poisson")
PHASES = ("all", "group", "KO")
TARGETS = tuple(f"entropy_{s}" for s in ("r32", "r16", "qf", "sf", "final", "winner"))
RQ1_P = [0.001, 0.004, 0.01, 0.02, 0.03, 0.04, 0.2, 0.5, 0.9]
RQ1_ADJUSTED = [0.009, 0.032, 0.07, 0.12, 0.15, 0.16, 0.6, 1.0, 1.0]


def _summary(rq: int) -> pd.DataFrame:
    if rq == 2:
        rows = [
            {"window": w, "grain": g, "model_name": m, "stage": s}
            for w, g, m, s in product(
                ("all", "placebo", "treated", "treated_unresolved"),
                ("cycle", "state"),
                (*MODELS, "mean_rate_poisson"),
                TARGETS,
            )
        ]
    else:
        metrics = ("rps", "nll", "rmse") if rq == 1 else ("ece", "rel", "bs", "res", "unc")
        rows = [
            {"phase_scheme": "group_ko", "phase": p, "model_name": m, "metric": k}
            for k, p, m in product(metrics, PHASES, MODELS)
        ]
        if rq == 1:
            rows += [
                {"phase_scheme": "group_r32_r16plus", "phase": p, "model_name": m, "metric": k}
                for k, p, m in product(metrics, ("all", "group", "R32", "R16+"), MODELS)
            ]
        else:
            rows += [
                {"phase_scheme": "period5", "phase": p, "model_name": m, "metric": k}
                for k, p, m in product(metrics, ("MD1", "MD2", "MD3", "R32", "R16+"), MODELS)
            ]
    table = pd.DataFrame(rows)
    table["p_perm"] = 1e-12  # excluded tiny p-values must never change core correction
    table["mean_diff"] = -0.012345
    table["ci_lo"] = -0.02
    table["ci_hi"] = 0.01
    if rq == 1:
        canonical = table["phase_scheme"] == "group_ko"
        table.loc[canonical & (table["metric"] == "rps"), "p_perm"] = RQ1_P
        table.loc[canonical & (table["metric"] == "nll"), "p_perm"] = 0.02
    elif rq == 2:
        core = (
            (table["window"] == "treated")
            & (table["grain"] == "state")
            & table["model_name"].isin(MODELS)
        )
        table.loc[core, "p_perm"] = [0.001, *([1.0] * 17)]
    else:
        canonical = table["phase_scheme"] == "group_ko"
        for metric, first in (("ece", 0.001), ("rel", 0.002)):
            table.loc[canonical & (table["metric"] == metric), "p_perm"] = [first, *([1.0] * 8)]
    return table


def _correct(table: pd.DataFrame, rq: int) -> pd.DataFrame:
    module = import_module(
        f"src.analysis.{('rq1_paired', 'rq2_paired', 'rq3_calibration_drift')[rq - 1]}"
    )
    return holm_core_families(table, module.HOLM_FAMILIES)


def test_rq1_separate_nine_test_families_and_no_duplicate_adjustment() -> None:
    table = _summary(1)
    result = _correct(table, 1)
    canonical = result["phase_scheme"] == "group_ko"
    assert result.loc[canonical & (result["metric"] == "rps"), "p_holm"].tolist() == pytest.approx(
        RQ1_ADJUSTED
    )
    assert result.loc[canonical & (result["metric"] == "nll"), "p_holm"].tolist() == pytest.approx(
        [0.18] * 9
    )
    assert result["p_holm"].notna().sum() == 18
    assert result.loc[~canonical | (result["metric"] == "rmse"), "p_holm"].isna().all()


def test_rq2_pools_six_targets_and_excludes_controls_windows_and_cycle_grain() -> None:
    result = _correct(_summary(2), 2)
    adjusted = result["p_holm"].dropna()
    assert len(adjusted) == 18
    assert adjusted.tolist() == pytest.approx([0.018, *([1.0] * 17)])
    core = (
        (result["window"] == "treated")
        & (result["grain"] == "state")
        & result["model_name"].isin(MODELS)
    )
    assert result.loc[~core, "p_holm"].isna().all()


def test_rq3_pools_ece_and_reliability_and_excludes_other_metrics_and_periods() -> None:
    result = _correct(_summary(3), 3)
    core = (result["phase_scheme"] == "group_ko") & result["metric"].isin(["ece", "rel"])
    assert result.loc[core, "p_holm"].notna().sum() == 18
    assert result.loc[core & (result["metric"] == "ece"), "p_holm"].iloc[0] == pytest.approx(0.018)
    assert result.loc[core & (result["metric"] == "rel"), "p_holm"].iloc[0] == pytest.approx(0.034)
    assert result.loc[~core, "p_holm"].isna().all()


@pytest.mark.parametrize("rq", [1, 2, 3])
def test_original_statistics_input_index_and_order_are_preserved(rq: int) -> None:
    table = _summary(rq).sample(frac=1, random_state=8)
    table.index = [5] * len(table)  # row labels must not affect membership or assignment
    original = table.copy(deep=True)
    result = _correct(table, rq)
    pd.testing.assert_frame_equal(table, original)
    pd.testing.assert_frame_equal(result.drop(columns="p_holm"), original)
    # Compare with unshuffled correction using the unique scientific row keys.
    keys = [c for c in table if c not in {"p_perm", "mean_diff", "ci_lo", "ci_hi"}]
    expected = _correct(_summary(rq), rq).set_index(keys)["p_holm"].sort_index()
    actual = result.set_index(keys)["p_holm"].sort_index()
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("rq", [1, 2, 3])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "missing_and_duplicate"])
def test_core_membership_must_cover_every_comparison_once(rq: int, defect: str) -> None:
    table = _summary(rq)
    first_core = _correct(table, rq)["p_holm"].first_valid_index()
    other_core = _correct(table, rq)["p_holm"].dropna().index[1]
    if defect in {"missing", "missing_and_duplicate"}:
        table = table.drop(index=first_core)
    if defect in {"duplicate", "missing_and_duplicate"}:
        table = pd.concat([table, table.loc[[other_core]]], ignore_index=True)
    with pytest.raises(ValueError, match="core comparisons"):
        _correct(table, rq)


@pytest.mark.parametrize("rq", [1, 2, 3])
@pytest.mark.parametrize("bad_p", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_core_p_value_is_an_error(rq: int, bad_p: float) -> None:
    table = _summary(rq)
    first_core = _correct(table, rq)["p_holm"].first_valid_index()
    table.loc[first_core, "p_perm"] = bad_p
    with pytest.raises(ValueError, match="p_perm"):
        _correct(table, rq)


@pytest.mark.parametrize("rq", [1, 2, 3])
def test_missing_selector_column_is_an_error(rq: int) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        _correct(_summary(rq).drop(columns="model_name"), rq)


@pytest.mark.parametrize("rq", [1, 2, 3])
def test_stage_cli_corrects_only_canonical_summary_and_preserves_descriptive_tests(
    rq: int,
    monkeypatch,
    tmp_path: Path,
) -> None:
    name = ("rq1_paired", "rq2_paired", "rq3_calibration_drift")[rq - 1]
    module = import_module(f"src.analysis.{name}")
    summary = _summary(rq)
    descriptive = pd.DataFrame({"p_perm": [np.nan, 0.0001], "mean_diff": [0.0, -0.1]})
    trend = pd.DataFrame({"p_frozen": [0.0001], "p_diff": [0.0002], "slope_diff": [0.01]})
    monkeypatch.setattr(module.pd, "read_csv", lambda _: pd.DataFrame())
    if rq == 1:
        monkeypatch.setattr(module, "pairing_coverage", lambda _: pd.DataFrame())
        monkeypatch.setattr(module, "paired_differences", lambda _: pd.DataFrame())
        core_builder, other_builder = "phase_table", "round_table"
        options = ("--phase-out", "--rounds-out", "--diffs-out", "--coverage-out")
    elif rq == 2:
        monkeypatch.setattr(module, "pair_cycles", lambda _: pd.DataFrame())
        monkeypatch.setattr(module, "mark_states", lambda x: x)
        monkeypatch.setattr(module, "pairing_coverage", lambda *_: pd.DataFrame())
        core_builder, other_builder = "stage_table", "matchday_table"
        options = ("--stage-out", "--matchday-out", "--diffs-out", "--coverage-out")
    else:
        monkeypatch.setattr(module, "load_paired", lambda _: pd.DataFrame())
        monkeypatch.setattr(module, "check_against_bins", lambda *_: None)
        monkeypatch.setattr(module, "trend_table", lambda *a, **kw: trend.copy())
        core_builder, other_builder = "phase_table", "round_table"
        options = ("--phase-out", "--rounds-out", "--trend-out")
    monkeypatch.setattr(module, core_builder, lambda *a, **kw: summary.copy())
    monkeypatch.setattr(module, other_builder, lambda *a, **kw: descriptive.copy())
    args = [
        part for i, option in enumerate(options) for part in (option, str(tmp_path / f"{i}.csv"))
    ]
    module.main(args)
    with (tmp_path / "0.csv").open() as handle:
        written = list(csv.DictReader(handle))
    expected = _correct(summary, rq)
    assert sum(row["p_holm"] != "" for row in written) == 18
    for original, adjusted, row in zip(
        summary.to_dict("records"), expected["p_holm"], written, strict=True
    ):
        for key, value in original.items():
            if isinstance(value, float):
                assert float(row[key]) == value
            else:
                assert row[key] == value
        if np.isfinite(adjusted):
            assert float(row["p_holm"]) == pytest.approx(adjusted)
        else:
            assert row["p_holm"] == ""
    with (tmp_path / "1.csv").open() as handle:
        assert all(row["p_holm"] == "" for row in csv.DictReader(handle))
    if rq == 3:
        with (tmp_path / "2.csv").open() as handle:
            row = next(csv.DictReader(handle))
        assert row == {"p_frozen": "0.0001", "p_diff": "0.0002", "slope_diff": "0.01", "p_holm": ""}
