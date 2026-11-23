"""Optional historical comparison; excluded from the current-results reproduction path."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PHASE_KEYS = ["phase_scheme", "phase", "model_name", "metric"]
TREND_KEYS = ["variant", "model_name", "metric"]


def compare_tables(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep every old/new estimate together and flag nominal threshold crossings."""
    result = old.merge(
        new,
        on=keys,
        how="outer",
        suffixes=("_original", "_90min"),
        validate="one_to_one",
        indicator=True,
    )
    if not (result._merge == "both").all():
        raise ValueError("Original and 90-minute tables have different hypothesis rows")
    result = result.drop(columns="_merge")
    for col in ("n", "n_matches", "n_periods"):
        if (
            f"{col}_original" in result
            and not (result[f"{col}_original"] == result[f"{col}_90min"]).all()
        ):
            raise ValueError(f"Original and 90-minute tables have different {col}")
    for col in ("p_perm", "p_holm", "p_frozen", "p_diff"):
        if f"{col}_original" in result:
            for version in ("original", "90min"):
                values = result[f"{col}_{version}"]
                result[f"{col}_lt_05_{version}"] = (values < 0.05).where(values.notna())
    return result


def _summary(row: pd.Series, suffix: str = "") -> str:
    return (
        f"Δ {row['mean_diff' + suffix]:+.6f}, 95% CI "
        f"[{row['ci_lo' + suffix]:+.6f}, {row['ci_hi' + suffix]:+.6f}], "
        f"raw p={row['p_perm' + suffix]:.4f}, Holm p={row['p_holm' + suffix]:.4f}"
    )


def write_comparison(
    baseline: Path,
    tables: Path,
    audit: pd.DataFrame,
    report: Path,
    *,
    comparison_tables: Path,
    n_boot: int,
    seed: int,
) -> None:
    comparison_tables.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    for name, keys in (
        ("rq1_paired_phase", PHASE_KEYS),
        ("rq3_calibration_phase", PHASE_KEYS),
        ("rq3_trend", TREND_KEYS),
    ):
        comparison = compare_tables(
            pd.read_csv(baseline / f"{name}.csv"), pd.read_csv(tables / f"{name}.csv"), keys
        )
        comparison.to_csv(
            comparison_tables / f"{name}_comparison.csv", index=False, float_format="%.9g"
        )
        comparisons[name] = comparison

    lines = [
        "# 90-minute rescoring: comparison for supervision",
        "",
        (
            "Separate results for discussion; the existing analysis remains available. "
            "These are the same saved forecasts scored against regulation time, including "
            "stoppage time, and excluding extra time and shootouts."
        ),
        "",
        "## What changed",
        "",
        (
            f"- Coverage: {len(audit)} matches. Goals change in {int(audit.goals_changed.sum())}; "
            f"home/draw/away labels change in {int(audit.outcome_changed.sum())}."
        ),
        (
            f"- Draws: {int((audit.actual_outcome == 1).sum())} under the original labels; "
            f"{int((audit.regulation_outcome == 1).sum())} after 90 minutes."
        ),
        (
            "- Forecasts, probabilities, fitted models, inference IDs, cadence pairing and "
            "MD1 ties are preserved. RQ2 is not changed by this rescoring."
        ),
        (
            f"- Existing tests and families are reused: {n_boot:,} resamples, seed {seed}; "
            "two-sided RQ1 and calibration-level tests; pointwise 95% bootstrap CIs. "
            "Holm families: 9 RPS, 9 NLL, 18 ECE/reliability tests. Exploratory trend tests "
            "remain one-sided and unadjusted."
        ),
        (
            "- The original pre-binned calibration file uses different labels and is not "
            "reused. RQ3 recomputes calibration from saved per-match probabilities and "
            "the 90-minute outcomes."
        ),
        "",
        "Changed match scores (original → 90 minutes):",
        "",
    ]
    for row in audit[audit.goals_changed].itertuples():
        lines.append(
            f"- {row.home}–{row.away} ({row.round_label}, {row.match_id}): "
            f"{row.actual_h}–{row.actual_a} → {row.regulation_h}–{row.regulation_a}; "
            f"outcome {'changes' if row.outcome_changed else 'unchanged'}."
        )

    lines += [
        "",
        "## RQ1: accuracy",
        "",
        "Differences are per-round minus frozen; negative favours refitting.",
        "",
    ]
    rq1 = comparisons["rq1_paired_phase"]
    core = rq1[rq1.phase_scheme.eq("group_ko") & rq1.metric.isin(["rps", "nll"])]
    for metric in ("rps", "nll"):
        cell = core[core.metric.eq(metric)]
        lines.append(
            f"- {metric.upper()}, 9 tests: raw p<0.05 "
            f"{int((cell.p_perm_original < 0.05).sum())} → {int((cell.p_perm_90min < 0.05).sum())}; "
            f"Holm p<0.05 {int((cell.p_holm_original < 0.05).sum())} → "
            f"{int((cell.p_holm_90min < 0.05).sum())}."
        )
    for _, row in core[core.phase.eq("all")].iterrows():
        lines += [
            (
                f"- Overall {row.model_name}, {row.metric.upper()}: "
                f"original {_summary(row, '_original')}; 90-minute {_summary(row, '_90min')}."
            )
        ]
    lines += ["", "Raw p<0.05 under the 90-minute labels (including phase subsets):", ""]
    for _, row in core[core.p_perm_90min < 0.05].iterrows():
        lines.append(
            f"- {row.model_name}, {row.phase}, {row.metric.upper()}: {_summary(row, '_90min')}."
        )
    if not (core.p_perm_90min < 0.05).any():
        lines.append("- None.")

    lines += [
        "",
        "## RQ3: calibration on the same matches",
        "",
        (
            "ECE and reliability use all/group/KO sets of 80/48/32 treated matches. "
            "Period views and trends remain exploratory. Negative level differences favour refitting."
        ),
        "",
    ]
    rq3 = comparisons["rq3_calibration_phase"]
    calibration = rq3[rq3.phase_scheme.eq("group_ko") & rq3.metric.isin(["ece", "rel"])]
    lines.append(
        f"- Core calibration, 18 tests: raw p<0.05 "
        f"{int((calibration.p_perm_original < 0.05).sum())} → {int((calibration.p_perm_90min < 0.05).sum())}; "
        f"Holm p<0.05 {int((calibration.p_holm_original < 0.05).sum())} → "
        f"{int((calibration.p_holm_90min < 0.05).sum())}."
    )
    for _, row in calibration[calibration.metric.eq("ece")].iterrows():
        lines.append(
            f"- {row.model_name}, {row.phase}, ECE: "
            f"{row.mean_frozen_90min:.6f} → {row.mean_per_round_90min:.6f}; "
            f"{_summary(row, '_90min')}."
        )
    lines += ["", "Tests crossing 0.05 after Holm in either version:", ""]
    flagged = calibration[(calibration.p_holm_original < 0.05) | (calibration.p_holm_90min < 0.05)]
    for _, row in flagged.iterrows():
        lines.append(
            f"- {row.model_name}, {row.phase}, {row.metric.upper()}: "
            f"original {_summary(row, '_original')}; 90-minute {_summary(row, '_90min')}."
        )
    if flagged.empty:
        lines.append("- None.")
    lines += [
        "",
        "## RQ3: exploratory trends",
        "",
        (
            "The rise test asks whether frozen calibration error increases over periods. "
            "The mitigation test asks whether the frozen slope is greater than the refitted slope. "
            "A mitigation result alone does not establish that frozen calibration deteriorated."
        ),
        "",
    ]
    for _, row in comparisons["rq3_trend"].iterrows():
        lines.append(
            f"- {row.model_name}, {row.variant}, {row.metric.upper()}: "
            f"frozen slope {row.slope_frozen_original:+.6f} → {row.slope_frozen_90min:+.6f}, "
            f"rise p {row.p_frozen_original:.4f} → {row.p_frozen_90min:.4f}; "
            f"slope gap {row.slope_diff_original:+.6f} → {row.slope_diff_90min:+.6f}, "
            f"mitigation p {row.p_diff_original:.4f} → {row.p_diff_90min:.4f}."
        )
    lines += [
        "",
        "## Reporting scope",
        "",
        (
            "The 90-minute results are the current reporting version. The original results "
            "are retained only to document the scoring correction. The same dependence and "
            "exchangeability limitations and pointwise CI interpretation remain; neither "
            "the tests nor the Holm families changed."
        ),
        "",
        "## Reproduction and files",
        "",
        (
            "The current analysis is reproduced with `dvc repro -s regulation_time`. "
            "This historical comparison is optional: `dvc repro -s regulation_time_comparison`. "
            "It reads the old tables and current tables and writes "
            "`docs/tables/regulation_time_comparison/` and this note. "
            "The main analysis and figures do not require the old result tables."
        ),
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(f"Wrote comparison to {report}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tables", type=Path, default=Path("docs/tables"))
    parser.add_argument("--tables", type=Path, default=Path("docs/tables/regulation_time"))
    parser.add_argument(
        "--audit", type=Path, default=Path("data/analysis/regulation_time/outcome_audit.csv")
    )
    parser.add_argument(
        "--comparison-tables", type=Path, default=Path("docs/tables/regulation_time_comparison")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/notes/regulation_time_comparison.md")
    )
    args = parser.parse_args()
    write_comparison(
        args.baseline_tables,
        args.tables,
        pd.read_csv(args.audit),
        args.report,
        comparison_tables=args.comparison_tables,
        n_boot=10000,
        seed=0,
    )


if __name__ == "__main__":
    main()
