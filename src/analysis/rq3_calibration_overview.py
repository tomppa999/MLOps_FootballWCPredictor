"""RQ3 overview: calibration in the group stage vs the knockout phase.

Pools the per-round reliability bins into the two tournament phases and reports
expected calibration error (ECE) per model and cadence mode, plus reliability
diagrams:

- ``calibration-group-vs-ko.png`` — all MODEL_STYLES models, both cadences
  (group includes MD1, matching the original figure).
- ``calibration-contrast-group-vs-ko.png`` — the three CONTRAST_MODELS only,
  2×2 (cadence × phase); group = MD2–MD3 so the placebo round is excluded.

Usage:
    python -m src.analysis.rq3_calibration_overview
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plot_entropy_trajectory import MODEL_STYLES
from src.analysis.rq1_paired import CONTRAST_MODELS
from src.analysis.rq_datasets.paths import RQ3_PATH

DEFAULT_OUTPUT: Final[Path] = Path("docs/figures/calibration-group-vs-ko.png")
CONTRAST_OUTPUT: Final[Path] = Path("docs/figures/calibration-contrast-group-vs-ko.png")

# Original pooled figure: MD1–3 vs KO (72 / 32).
GROUP_ROUNDS: Final[frozenset[str]] = frozenset({"1", "2", "3"})
# Contrast figure / RQ3 analysis: MD1 is placebo, so group = MD2–3 (48 / 32).
TREATED_GROUP_ROUNDS: Final[frozenset[str]] = frozenset({"2", "3"})
PHASES: Final[tuple[str, ...]] = ("group", "knockout")
CADENCES: Final[tuple[str, ...]] = ("frozen", "per_round")


def load_reliability(path: Path = RQ3_PATH) -> pd.DataFrame:
    """Read reliability bins (no phase label yet — callers choose the grouping)."""
    df = pd.read_csv(path)
    return df[df["record_type"] == "reliability"].copy()


def assign_phase(df: pd.DataFrame, group_rounds: frozenset[str]) -> pd.DataFrame:
    """Label rows with tournament phase given a group-round set."""
    out = df.copy()
    out["phase"] = out["round_label"].astype(str).apply(
        lambda label: "group" if label in group_rounds else "knockout",
    )
    # Drop rounds that belong to neither (e.g. MD1 when group = MD2–3).
    keep = out["round_label"].astype(str).isin(group_rounds) | (
        ~out["round_label"].astype(str).isin(GROUP_ROUNDS)
    )
    return out.loc[keep]


def pool_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rounds into phases, weighting each bin by its match count.

    ``observed_freq`` is a per-round rate, so it has to be turned back into an
    event count before summing, otherwise rounds with one match in a bin would
    carry the same weight as rounds with twenty.
    """
    df = df.copy()
    df["events"] = df["observed_freq"] * df["n"]
    df["predicted_mass"] = df["mean_predicted_p"] * df["n"]

    grouped = df.groupby(
        ["cadence_mode", "model_name", "phase", "outcome_class", "bin_index"],
        as_index=False,
    ).agg(n=("n", "sum"), events=("events", "sum"), predicted_mass=("predicted_mass", "sum"))

    grouped = grouped[grouped["n"] > 0]
    grouped["observed_freq"] = grouped["events"] / grouped["n"]
    grouped["mean_predicted_p"] = grouped["predicted_mass"] / grouped["n"]
    return grouped


def expected_calibration_error(pooled: pd.DataFrame) -> pd.DataFrame:
    """ECE per model, cadence and phase, pooled over the three outcome classes."""
    pooled = pooled.copy()
    pooled["abs_gap_mass"] = (
        pooled["observed_freq"] - pooled["mean_predicted_p"]
    ).abs() * pooled["n"]

    ece = pooled.groupby(["model_name", "cadence_mode", "phase"], as_index=False).apply(
        lambda g: pd.Series({"ece": g["abs_gap_mass"].sum() / g["n"].sum()}),
        include_groups=False,
    )
    return ece.pivot_table(index="model_name", columns=["phase", "cadence_mode"], values="ece")


def _ece_scalar(pooled: pd.DataFrame, model: str, mode: str, phase: str) -> float:
    series = pooled[
        (pooled["model_name"] == model)
        & (pooled["cadence_mode"] == mode)
        & (pooled["phase"] == phase)
    ]
    if series.empty or series["n"].sum() == 0:
        return float("nan")
    abs_gap = (series["observed_freq"] - series["mean_predicted_p"]).abs() * series["n"]
    return float(abs_gap.sum() / series["n"].sum())


def _curve(series: pd.DataFrame) -> pd.DataFrame:
    """Pool the three outcome classes into one (predicted, observed, n) per bin."""
    return (
        series.groupby("bin_index")
        .apply(
            lambda g: pd.Series({
                "predicted": (g["mean_predicted_p"] * g["n"]).sum() / g["n"].sum(),
                "observed": g["events"].sum() / g["n"].sum(),
                "n": g["n"].sum(),
            }),
            include_groups=False,
        )
        .sort_index()
    )


def plot_reliability(pooled: pd.DataFrame) -> plt.Figure:
    """Reliability diagram per phase: predicted vs observed, both cadences."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)

    for ax, phase in zip(axes, PHASES, strict=True):
        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1.0, zorder=1)

        for model, style in MODEL_STYLES.items():
            for mode in CADENCES:
                if model == "mean_rate_poisson" and mode == "frozen":
                    continue

                series = pooled[
                    (pooled["model_name"] == model)
                    & (pooled["cadence_mode"] == mode)
                    & (pooled["phase"] == phase)
                ]
                if series.empty:
                    continue

                curve = _curve(series)
                ax.plot(
                    curve["predicted"],
                    curve["observed"],
                    color=style[mode],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.4 if mode == "per_round" else 1.1,
                    linestyle="-" if mode == "per_round" else "--",
                    zorder=3 if mode == "per_round" else 2,
                )

        n_matches = 72 if phase == "group" else 32
        ax.set_title(f"{phase.capitalize()} stage (n = {n_matches} matches)")
        ax.set_xlabel("Mean predicted probability")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    axes[0].set_ylabel("Observed frequency")

    handles = [
        plt.Line2D(
            [], [],
            color=style[mode],
            linestyle="-" if mode == "per_round" else "--",
            marker="o",
            markersize=3.5,
            label=(
                style["label"]
                if model == "mean_rate_poisson"
                else f"{style['label']} ({mode.replace('_', '-')})"
            ),
        )
        for model, style in MODEL_STYLES.items()
        for mode in CADENCES
        if not (model == "mean_rate_poisson" and mode == "frozen")
    ]
    handles.append(
        plt.Line2D([], [], color="0.6", linestyle=":", label="Perfect calibration"),
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)

    fig.suptitle(
        "RQ3: calibration by tournament phase, frozen vs per-round retraining\n"
        "(outcome classes pooled; bins weighted by match count)",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return fig


def plot_contrast_reliability(pooled: pd.DataFrame) -> plt.Figure:
    """2×2 reliability diagram: rows = cadence, columns = group (MD2–3) / KO.

    Restricted to ``CONTRAST_MODELS``. Marker area ∝ bin mass; ECE annotated
    in each panel's legend entry.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    n_by_phase = {"group": 48, "knockout": 32}

    for row, mode in enumerate(CADENCES):
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1.0, zorder=1)

            for model in CONTRAST_MODELS:
                style = MODEL_STYLES[model]
                series = pooled[
                    (pooled["model_name"] == model)
                    & (pooled["cadence_mode"] == mode)
                    & (pooled["phase"] == phase)
                ]
                if series.empty:
                    continue
                curve = _curve(series)
                ece = _ece_scalar(pooled, model, mode, phase)
                sizes = 20.0 + 4.0 * np.sqrt(curve["n"].to_numpy(dtype=float))
                ax.plot(
                    curve["predicted"],
                    curve["observed"],
                    color=style[mode],
                    linewidth=1.5,
                    linestyle="-" if mode == "per_round" else "--",
                    zorder=2,
                    label=f"{style['label']} (ECE={ece:.3f})",
                )
                ax.scatter(
                    curve["predicted"],
                    curve["observed"],
                    s=sizes,
                    color=style[mode],
                    zorder=3,
                    alpha=0.85,
                )

            ax.set_title(
                f"{mode.replace('_', '-')} · {phase} (n = {n_by_phase[phase]})"
            )
            ax.grid(alpha=0.25)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="upper left", frameon=False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency")

    fig.suptitle(
        "RQ3: contrast-model calibration, group (MD2–3) vs knockout\n"
        "(marker area ∝ bin mass; MD1 placebo excluded)",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RQ3_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contrast-output", type=Path, default=CONTRAST_OUTPUT)
    args = parser.parse_args()

    reliability = load_reliability(args.input)

    # Original all-roster figure (group includes MD1).
    pooled_all = pool_bins(assign_phase(reliability, GROUP_ROUNDS))
    ece = expected_calibration_error(pooled_all)
    print("Expected calibration error — all rounds in group (incl. MD1)\n")
    print(ece.round(4).to_string())

    roster = [m for m in MODEL_STYLES if m in ece.index]
    print("\nSelected roster only, per-round minus frozen (negative = per-round better)\n")
    for phase in PHASES:
        delta = ece[(phase, "per_round")] - ece[(phase, "frozen")]
        print(f"  {phase:9s} " + "  ".join(f"{m}={delta[m]:+.4f}" for m in roster))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_reliability(pooled_all[pooled_all["model_name"].isin(MODEL_STYLES)])
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(f"\nSaved figure to {args.output}")

    # Contrast-only figure (group = MD2–3).
    pooled_contrast = pool_bins(assign_phase(reliability, TREATED_GROUP_ROUNDS))
    pooled_contrast = pooled_contrast[pooled_contrast["model_name"].isin(CONTRAST_MODELS)]
    ece_c = expected_calibration_error(pooled_contrast)
    print("\nContrast models only, group = MD2–3 (MD1 excluded)\n")
    print(ece_c.round(4).to_string())

    contrast_fig = plot_contrast_reliability(pooled_contrast)
    args.contrast_output.parent.mkdir(parents=True, exist_ok=True)
    contrast_fig.savefig(args.contrast_output, dpi=200)
    plt.close(contrast_fig)
    print(f"Saved contrast figure to {args.contrast_output}")


if __name__ == "__main__":
    main()
