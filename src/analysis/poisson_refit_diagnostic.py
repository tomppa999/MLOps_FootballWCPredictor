"""Isolated MD1 refit diagnostics; never replace a registered or reconstructed model.

Run through ``dvc repro -s poisson_refit_diagnostic``. Historical pickle files
are scanned for numeric array payloads only; they are never deserialized.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickletools
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import OptimizeResult, approx_fprime, minimize
from sklearn.preprocessing import StandardScaler

from src.models.candidates.poisson_glm import BivariatePoisson, _bvp_log_pmf
from src.models.config import MODEL_FEATURE_SETS
from src.models.data_split import make_splits

FEATURES = MODEL_FEATURE_SETS["poisson_glm"]
ALPHA = 9.909387789987418
GOLD_COMMITS = {
    "pre": "d1538a5fe65e61136bfd30cbe16f012b118c6cb7",
    "post": "b9a7de646e53d92199be054a75734ef9a6e8ff8d",
}
MODEL_HASHES = {
    88: "dc0cf6bb8b4766610f61ce25caa6abd46c195d74f50f7f5fc16fbb30b9aed6a8",
    95: "93b4cf980708b80756514b74eb60fc06d0270b6a29d24b8292d933ce9a32428f",
    98: "209805bf2023a04852a1808761f10387af2ae8c43a2fabca8f9a914a1c5878da",
}
OUT = Path("docs/tables/poisson_refit_diagnostic")
REPORT = Path("docs/notes/poisson_refit_diagnostic.md")


def numeric_state(path: Path, *, n_features: int = 14) -> dict[str, np.ndarray]:
    """Extract fixed-size float64 payloads without executing pickle instructions."""
    fields = {"mean_": n_features, "scale_": n_features}
    fields.update({f"_coef{i}": n_features + 1 for i in (1, 2, 3)})
    state: dict[str, np.ndarray] = {}
    key = None
    for op, arg, _position in pickletools.genops(path.read_bytes()):
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE"}:
            if arg in fields or arg == "var_":
                key = arg if arg in fields else None
        elif (
            op.name in {"BYTEARRAY8", "SHORT_BINBYTES", "BINBYTES"}
            and key is not None and len(arg) == 8 * fields[key]
        ):
            if key in state:
                raise ValueError(f"Duplicate numeric field {key}")
            state[key] = np.frombuffer(arg, dtype="<f8").copy()
    missing = fields.keys() - state.keys()
    if missing:
        raise ValueError(f"Missing numeric fields: {sorted(missing)}")
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError("Non-finite numeric state")
    if (state["scale_"] <= 0).any():
        raise ValueError("Invalid scaler scale")
    return state


def rebase_params(
    state: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray,
) -> np.ndarray:
    """Express historical coefficients in a new scaler without changing rates."""
    blocks = []
    for i in (1, 2, 3):
        old = state[f"_coef{i}"]
        intercept = old[0] + ((mean - state["mean_"]) / state["scale_"]) @ old[1:]
        blocks.append(np.r_[intercept, old[1:] * scale / state["scale_"]])
    return np.concatenate(blocks)


def fit_with_start(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, *,
    start: dict[str, np.ndarray] | None, alpha: float, maxiter: int = 500,
    strict: bool = False,
) -> tuple[BivariatePoisson, OptimizeResult]:
    """Use the production objective, exposing solver status and optional warm starts."""
    model = BivariatePoisson(alpha=alpha, maxiter=maxiter)
    model._scaler = StandardScaler().fit(x)
    model._n_features = x.shape[1]
    design = model._build_design(model._scaler.transform(x))
    weights = weights * (len(weights) / weights.sum())
    initial = (
        np.zeros(3 * (x.shape[1] + 1)) if start is None
        else rebase_params(start, model._scaler.mean_, model._scaler.scale_)
    )
    options = {"maxiter": maxiter, "ftol": 1e-9}
    if strict:
        options.update(ftol=1e-12, gtol=1e-7, maxfun=100000)
    result = minimize(
        model._neg_log_likelihood, initial,
        args=(design, y[:, 0].astype(float), y[:, 1].astype(float), weights),
        method="L-BFGS-B", options=options,
    )
    # Keep a failed solver's last iterate for inspection, rather than silently
    # replacing it with production's zero-coefficient fallback.
    model._coef1, model._coef2, model._coef3 = model._unpack(result.x)
    return model, result


def _load_numeric_model(version: int) -> dict[str, np.ndarray]:
    path = Path(f"data/reconstruction/inputs/models/wc_shadow/v{version}/python_model.pkl")
    if not path.exists():
        cache = Path(os.environ.get(
            "WC_POISSON_DIAGNOSTIC_MODEL_CACHE", str(Path(tempfile.gettempdir()) / "wc-rq2-static-models"),
        ))
        path = cache / f"v{version}" / "python_model.pkl"
        if not path.exists():
            import mlflow

            from src.models.mlflow_utils import setup_mlflow

            setup_mlflow()
            root = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/wc_shadow/{version}", dst_path=str(cache / f"v{version}"),
            )
            path = Path(root) / "python_model.pkl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_HASHES[version]:
        raise ValueError(f"Historical model v{version} failed SHA256 validation")
    return numeric_state(path)


def _historical_gold(commit: str) -> tuple[pd.DataFrame, dict]:
    lock = yaml.safe_load(subprocess.check_output(["git", "show", f"{commit}:dvc.lock"], text=True))
    directory_hash = lock["stages"]["gold"]["outs"][0]["md5"]
    cache = Path(".dvc/cache/files/md5")
    directory = cache / directory_hash[:2] / directory_hash[2:]
    if directory.exists():
        entries = json.loads(directory.read_text())
        file_hash = next(row["md5"] for row in entries if row["relpath"] == "matches.parquet")
        path = cache / file_hash[:2] / file_hash[2:]
        if path.exists():
            if hashlib.md5(path.read_bytes()).hexdigest() != file_hash:
                raise ValueError(f"Historical Gold hash mismatch at {commit}")
            return pd.read_parquet(path), {"commit": commit, "directory_md5": directory_hash, "file_md5": file_hash}
    # Read-only materialisation of a historical DVC version; no repro or pull
    # into the working tree, matching the reconstruction's historical reader.
    with tempfile.TemporaryDirectory(prefix="poisson-diagnostic-gold-") as temp:
        subprocess.run(
            ["dvc", "get", ".", "data/gold", "--rev", commit, "-o", f"{temp}/gold"],
            check=True, capture_output=True,
        )
        frame = pd.read_parquet(f"{temp}/gold/matches.parquet")
    return frame, {"commit": commit, "directory_md5": directory_hash}


def _state(model: BivariatePoisson) -> dict[str, np.ndarray]:
    return {
        "mean_": model._scaler.mean_, "scale_": model._scaler.scale_,
        **{f"_coef{i}": getattr(model, f"_coef{i}") for i in (1, 2, 3)},
    }


def _components(state: dict[str, np.ndarray], x: np.ndarray) -> list[np.ndarray]:
    design = np.column_stack([np.ones(len(x)), (x - state["mean_"]) / state["scale_"]])
    return [np.exp(design @ state[f"_coef{i}"]).clip(1e-6) for i in (1, 2, 3)]


def _evaluate(state: dict[str, np.ndarray], split, serving: np.ndarray) -> dict:
    weights = split.w_full * (len(split.w_full) / split.w_full.sum())
    model = BivariatePoisson(alpha=ALPHA)
    model._n_features = len(FEATURES)
    scaler = StandardScaler().fit(split.X_full)
    params = rebase_params(state, scaler.mean_, scaler.scale_)
    design = model._build_design(scaler.transform(split.X_full))
    args = (design, split.y_full[:, 0], split.y_full[:, 1], weights)
    gradient = approx_fprime(params, model._neg_log_likelihood, 1e-8, *args)
    l1, l2, l3 = _components(state, split.X_full)
    losses = -_bvp_log_pmf(split.y_full[:, 0], split.y_full[:, 1], l1, l2, l3) * weights
    md1 = (split.df_full.league_id == 1) & (split.df_full.season == 2026)
    s1, s2, s3 = _components(state, serving)
    return {
        "objective": float(model._neg_log_likelihood(params, *args)),
        "gradient_max_abs": float(np.abs(gradient).max()),
        "old_rows_weighted_nll": float(losses[~md1].sum()),
        "md1_weighted_nll": float(losses[md1].sum()),
        "serving_lambda1_mean": float(s1.mean()),
        "serving_lambda2_mean": float(s2.mean()),
        "serving_lambda3_mean": float(s3.mean()),
        "difference_variance_inflation": float((s1 + s2 + 2 * s3).mean() / (s1 + s2).mean()),
    }


def main() -> None:
    from src.analysis.replay_common import (
        augment_gold_for_inference,
        parse_wc_results_before_kickoff,
    )
    from src.inference.features import build_inference_features, generate_all_wc_pairings

    OUT.mkdir(parents=True, exist_ok=True)
    historical = {version: _load_numeric_model(version) for version in (88, 95, 98)}
    gold, inputs, splits = {}, {}, {}
    for period, commit in GOLD_COMMITS.items():
        gold[period], inputs[period] = _historical_gold(commit)
        splits[period] = make_splits(gold[period], FEATURES, half_period_years=3.0)
    pre, post = splits["pre"], splits["post"]
    if (len(pre.X_full), len(post.X_full)) != (6677, 6701):
        raise ValueError("Historical training row counts do not match registry versions 88/98")
    indexed_pre = pre.df_full.set_index("fixture_id")
    indexed_post = post.df_full.set_index("fixture_id")
    old_ids = indexed_pre.index.intersection(indexed_post.index)
    changed = indexed_pre.loc[old_ids, FEATURES + ["home_goals", "away_goals"]].compare(
        indexed_post.loc[old_ids, FEATURES + ["home_goals", "away_goals"]],
    )
    new_rows = indexed_post.loc[indexed_post.index.difference(indexed_pre.index)]
    if len(new_rows) != 24 or not ((new_rows.league_id == 1) & (new_rows.season == 2026)).all():
        raise ValueError("Training delta is not exactly the 24 MD1 World Cup fixtures")
    if new_rows.date_utc.max() >= pd.Timestamp("2026-06-18 14:09:14"):
        raise ValueError("A training fixture is not earlier than the observed inference cycle")

    wc = parse_wc_results_before_kickoff(pd.Timestamp("2026-06-18 14:09:14", tz="UTC"))
    augmented, date = augment_gold_for_inference(gold["post"], wc)
    features = build_inference_features(generate_all_wc_pairings(reference_date=date), augmented)
    hosts = {"Canada", "Mexico", "United States"}
    features = features[~features.home_team.isin(hosts) & ~features.away_team.isin(hosts)]
    serving = features[FEATURES].to_numpy(dtype=float)
    # Require exact reproduction of both logged regimes before interpreting
    # extracted coefficients or any diagnostic optimisation.
    logged = pd.read_parquet(
        "data/reconstruction/inputs/logged_predictions/all_models.parquet",
        filters=[("model_name", "==", "poisson_glm"), ("inference_run_id", "in", [
            "d92591c2b4d7452facacc351a69eacf4", "71888401cf044e66bfc49457bfdcac48",
        ])],
    )
    for version, run in [(95, "d92591c2b4d7452facacc351a69eacf4"), (98, "71888401cf044e66bfc49457bfdcac48")]:
        l1, l2, l3 = _components(historical[version], serving)
        predicted = features[["home_team", "away_team"]].copy()
        predicted["h"], predicted["a"] = l1 + l3, l2 + l3
        joined = predicted.merge(logged[logged.inference_run_id == run], on=["home_team", "away_team"])
        if len(joined) != 990:
            raise ValueError("Expected 990 verified non-host pairings")
        np.testing.assert_allclose(joined[["h", "a"]], joined[["lambda_h", "lambda_a"]], atol=1e-12, rtol=0)

    rows = []
    fitted_states = {}
    for version, state in historical.items():
        for period, split in splits.items():
            rows.append({"fit": f"historical_v{version}", "period": period, **_evaluate(state, split, serving)})
    # A zero start reproduces the original procedure. Cross-starts ask whether
    # each dataset supports both basins or pulls either start to one solution.
    schedule = [
        ("pre_zero", "pre", None, 500),
        ("post_zero", "post", None, 500),
        ("post_from_frozen", "post", historical[88], 2000),
        ("post_from_refit", "post", historical[98], 2000),
        ("pre_from_refit", "pre", historical[98], 2000),
    ]
    for name, period, start, maxiter in schedule:
        print(f"Fitting {name} ({len(splits[period].X_full)} rows)", flush=True)
        split = splits[period]
        began = time.monotonic()
        model, result = fit_with_start(
            split.X_full, split.y_full, split.w_full, start=start, alpha=ALPHA, maxiter=maxiter,
        )
        row = {
            "fit": name, "period": period, "success": bool(result.success),
            "message": str(result.message), "iterations": int(result.nit),
            "evaluations": int(result.nfev), "seconds": time.monotonic() - began,
            **_evaluate(_state(model), split, serving),
        }
        rows.append(row)
        fitted_states[name] = _state(model)
        print(json.dumps(row), flush=True)
    # Relative objective stopping can report success on a shallow surface.
    # Refine both pre-tournament solutions and the post-MD1 solution before
    # attributing different shared rates to a hard iteration cutoff.
    for initial in ("pre_zero", "pre_from_refit", "post_from_frozen"):
        period = "pre" if initial.startswith("pre_") else "post"
        split = splits[period]
        print(f"Strict refinement of {initial}", flush=True)
        began = time.monotonic()
        model, result = fit_with_start(
            split.X_full, split.y_full, split.w_full,
            start=fitted_states[initial], alpha=ALPHA, maxiter=2000, strict=True,
        )
        row = {
            "fit": f"strict_{initial}", "period": period, "success": bool(result.success),
            "message": str(result.message), "iterations": int(result.nit),
            "evaluations": int(result.nfev), "seconds": time.monotonic() - began,
            **_evaluate(_state(model), split, serving),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    fits = pd.DataFrame(rows)
    fits.to_csv(OUT / "fits.csv", index=False)
    weights = post.w_full * (len(post.w_full) / post.w_full.sum())
    md1 = (post.df_full.league_id == 1) & (post.df_full.season == 2026)
    tier1 = post.df_full.competition_tier == 1
    neutral_group = tier1 & post.df_full.is_neutral & ~post.df_full.is_knockout
    metadata = {
        "gold": inputs, "model_sha256": MODEL_HASHES, "features": FEATURES,
        "alpha": ALPHA, "half_period_years": 3.0,
        "pre_rows": len(pre.X_full), "post_rows": len(post.X_full),
        "old_rows_changed_in_model_inputs": len(changed),
        "new_rows": len(new_rows), "new_rows_weight_share": float(weights[md1].sum() / weights.sum()),
        "new_rows_mean_weight": float(weights[md1].mean()),
        "tier1_training_rows": int(tier1.sum()),
        "new_rows_share_tier1_weight": float(weights[md1 & tier1].sum() / weights[tier1].sum()),
        "neutral_tier1_group_rows": int(neutral_group.sum()),
        "md1_share_neutral_tier1_group_weight": float(
            weights[md1 & neutral_group].sum() / weights[neutral_group].sum(),
        ),
        "serving_pairings": len(serving), "logged_rate_match_atol": 1e-12,
        "scope": "Diagnostic fits only; no production model or reconstructed dataset is replaced.",
    }
    (OUT / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    new_rows.to_csv(OUT / "added_md1_rows.csv")
    terms = []
    for version in (88, 95, 98):
        state = historical[version]
        contribution = ((serving - state["mean_"]) / state["scale_"]) * state["_coef3"][1:]
        for feature, value in zip(["intercept", *FEATURES], np.r_[state["_coef3"][0], contribution.mean(axis=0)]):
            terms.append({"version": version, "feature": feature, "mean_log_lambda3_contribution": value})
    pd.DataFrame(terms).to_csv(OUT / "shared_component_terms.csv", index=False)
    lines = [
        "# MD1 Poisson refit diagnostic", "", "Generated by `dvc repro -s poisson_refit_diagnostic`.", "",
        "## Input checks", "",
        f"- Training rows: {len(pre.X_full)} frozen to {len(post.X_full)} after MD1; exactly 24 added WC fixtures.",
        f"- Existing rows with changed model features or targets: {len(changed)}.",
        f"- Added fixtures represent {100 * metadata['new_rows_weight_share']:.3f}% of normalized training weight.",
        f"- Within tier-1 training matches they represent {100 * metadata['new_rows_share_tier1_weight']:.3f}% of weight.",
        f"- Within neutral tier-1 group matches MD1 represents {100 * metadata['md1_share_neutral_tier1_group_weight']:.3f}% of weight.",
        "- Numeric historical states reproduce the logged v95/v98 rates on all 990 non-host pairs within 1e-12.",
        "- Model artifacts were inspected statically, without unpickling or executing downloaded code.", "",
        "## Optimizer checks", "",
    ]
    for row in rows:
        lines.append(
            f"- {row['fit']} on {row['period']}: penalized objective {row['objective']:.6f}; "
            f"max absolute finite-difference gradient {row['gradient_max_abs']:.5g}; "
            f"mean serving lambda3 {row['serving_lambda3_mean']:.6f}" +
            (f"; solver success={row['success']}, {row['message']}" if "success" in row else "") + "."
        )
    lines += [
        "", "## Scope", "",
        "Objective values are comparable only within the same training dataset. Warm-start coefficients are rebased to the new scaler before optimization.",
        "The serving mean covers 990 hypothetical pairings without hosts at the MD1 boundary; it is not a mean over actual played matches.",
        "Solver success alone does not establish a global optimum or identify the shared component uniquely. These checks compare specified starts, not an exhaustive search.",
        "No production models, historical predictions, RQ inputs, or existing RQ outputs were replaced.", "",
    ]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
