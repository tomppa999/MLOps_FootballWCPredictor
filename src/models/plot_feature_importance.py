"""Generate a feature importance heatmap from MLflow experimental runs.

Usage:
    python -m src.models.plot_feature_importance [--output plots/feature_importance.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

from src.models.mlflow_utils import EXPERIMENT_NAME, setup_mlflow


def load_importance_tables() -> dict[str, pd.DataFrame]:
    """Download importance CSVs from the latest experimental 'best' runs."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.role = 'best' AND tags.stage = 'experimental'",
        order_by=["start_time DESC"],
    )

    tables: dict[str, pd.DataFrame] = {}
    seen: set[str] = set()
    for run in runs:
        model_name = run.data.tags.get("model_name", "unknown")
        if model_name in seen:
            continue
        seen.add(model_name)
        artifact_dir = client.download_artifacts(run.info.run_id, "importance")
        csv_file = Path(artifact_dir) / f"importance_{model_name}.csv"
        if csv_file.exists():
            tables[model_name] = pd.read_csv(csv_file)

    return tables


def build_heatmap_matrix(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Pivot per-model importance tables into a (features x models) matrix."""
    all_features: set[str] = set()
    for df in tables.values():
        all_features.update(df["feature"].tolist())

    model_names = sorted(tables.keys())
    feature_list = sorted(all_features)

    matrix = pd.DataFrame(0.0, index=feature_list, columns=model_names)
    for name, df in tables.items():
        for _, row in df.iterrows():
            matrix.loc[row["feature"], name] = row["importance_mean"]

    row_max = matrix.max(axis=1)
    matrix = matrix.loc[row_max.sort_values(ascending=False).index]
    return matrix


def plot_heatmap(matrix: pd.DataFrame, output: Path) -> None:
    """Render and save the importance heatmap."""
    n_features = len(matrix)
    n_models = len(matrix.columns)
    fig, ax = plt.subplots(
        figsize=(max(10, n_models * 1.4), max(8, n_features * 0.45)),
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Δ RPS (permutation importance)"},
    )
    ax.set_title("Permutation Feature Importance by Model", fontsize=14)
    ax.set_ylabel("Feature")
    ax.set_xlabel("Model")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature importance heatmap from MLflow artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/feature_importance.png"),
        help="Output path for the heatmap image (default: plots/feature_importance.png)",
    )
    args = parser.parse_args()

    tables = load_importance_tables()
    if not tables:
        print("No importance artifacts found in MLflow. Run the pipeline first.")
        return

    print(f"Loaded importance data for {len(tables)} models: {sorted(tables.keys())}")
    matrix = build_heatmap_matrix(tables)
    plot_heatmap(matrix, args.output)


if __name__ == "__main__":
    main()
