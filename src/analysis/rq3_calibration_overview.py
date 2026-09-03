"""RQ3 overview: calibration in the group stage vs the knockout phase.

Pools the per-round reliability bins into the two tournament phases and reports
expected calibration error (ECE) per model and cadence mode, plus a reliability
diagram for each phase.

Usage:
    python -m src.analysis.rq3_calibration_overview [--output docs/figures/calibration-group-vs-ko.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.plot_entropy_trajectory import MODEL_STYLES
from src.analysis.rq_datasets.paths import RQ3_PATH

DEFAULT_OUTPUT: Final[Path] = Path("docs/figures/calibration-group-vs-ko.png")

GROUP_ROUNDS: Final[frozenset[str]] = frozenset({"1", "2", "3"})
PHASES: Final[tuple[str, ...]] = ("group", "knockout")


def load_reliability(path: Path = RQ3_PATH) -> pd.DataFrame:
    """Read reliability bins and label each row with its tournament phase."""
    df = pd.read_csv(path)
    df = df[df["record_type"] == "reliability"].copy()
    df["phase"] = df["round_label"].apply(
        lambda label: "group" if label in GROUP_ROUNDS else "knockout",
    )
    return df


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


def plot_reliability(pooled: pd.DataFrame) -> plt.Figure:
    """Reliability diagram per phase: predicted vs observed, both cadences."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)

    for ax, phase in zip(axes, PHASES, strict=True):
        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1.0, zorder=1)

        for model, style in MODEL_STYLES.items():
            for mode in ("frozen", "per_round"):
                if model == "mean_rate_poisson" and mode == "frozen":
                    continue

                series = pooled[
                    (pooled["model_name"] == model)
                    & (pooled["cadence_mode"] == mode)
                    & (pooled["phase"] == phase)
                ]
                if series.empty:
                    continue

                # Pool the three outcome classes into one curve per bin.
                curve = series.groupby("bin_index").apply(
                    lambda g: pd.Series({
                        "predicted": (g["mean_predicted_p"] * g["n"]).sum() / g["n"].sum(),
                        "observed": g["events"].sum() / g["n"].sum(),
                    }),
                    include_groups=False,
                ).sort_index()

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
        for mode in ("frozen", "per_round")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RQ3_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reliability = load_reliability(args.input)
    pooled = pool_bins(reliability)

    ece = expected_calibration_error(pooled)
    print("Expected calibration error (lower = better calibrated)\n")
    print(ece.round(4).to_string())

    roster = [m for m in MODEL_STYLES if m in ece.index]
    print("\nSelected roster only, per-round minus frozen (negative = per-round better)\n")
    for phase in PHASES:
        delta = ece[(phase, "per_round")] - ece[(phase, "frozen")]
        print(f"  {phase:9s} " + "  ".join(f"{m}={delta[m]:+.4f}" for m in roster))

    figure = plot_reliability(pooled[pooled["model_name"].isin(MODEL_STYLES)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(f"\nSaved figure to {args.output}")


if __name__ == "__main__":
    main()
