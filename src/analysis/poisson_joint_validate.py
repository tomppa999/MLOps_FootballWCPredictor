"""Verify the Poisson replacements against pinned match, outage and simulation evidence."""

from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.analysis import replay_common as rc
from src.analysis.poisson_joint_prepare import (
    BUNDLE,
    FEATURES,
    MEANS,
    MODEL,
    PINS,
    PREP,
    PROBS,
    SHARED,
    SIDES,
    TOL,
    poiss_rows,
)
from src.analysis.poisson_joint_replay import OUT, simulate_state
from src.analysis.rq2_outages import SNAPSHOTS, public_results
from src.analysis.strand4_entropy import build_entropy_snapshot_specs
from src.inference.features import build_inference_features, generate_all_wc_pairings
from src.models.candidates.poisson_glm import BivariatePoisson
from src.models.data_split import load_gold
from src.models.evaluation import compute_outcome_probs, compute_rps

CHECKS = Path("data/analysis/poisson_joint_checks.json")
TRIAL = Path("data/thesis_inputs/sensitivity.tar.gz")


def check_matches():
    keys = ["match_id", "cadence_mode", "model_name"]
    baseline = poiss_rows("rq1_matches.csv", keys).set_index(keys).sort_index()
    corrected = (
        pd.read_csv(OUT / "matches.csv", float_precision="round_trip")
        .set_index(keys, verify_integrity=True)
        .sort_index()
    )
    pd.testing.assert_index_equal(baseline.index, corrected.index)
    pd.testing.assert_frame_equal(baseline[MEANS], corrected[MEANS], check_exact=True)
    for col in ["provenance", "inference_run_id", "source_run_id"]:
        pd.testing.assert_series_equal(baseline[col], corrected[col])
    p = corrected[PROBS].to_numpy()
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid outcome probability")
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=TOL)
    np.testing.assert_allclose(
        compute_rps(p, baseline.actual_outcome.to_numpy()), corrected.rps, atol=TOL, rtol=0
    )
    np.testing.assert_allclose(
        compute_outcome_probs(baseline.lambda_h, baseline.lambda_a),
        baseline[PROBS],
        rtol=0,
        atol=TOL,
    )
    md1 = baseline.index.get_level_values("match_id").isin(
        baseline.loc[baseline.round_label == "1"].index.get_level_values("match_id")
    )
    twins = corrected.loc[md1, [*MEANS, SHARED, *PROBS, "rps", "version"]]
    frozen, refit = (twins.xs(arm, level="cadence_mode") for arm in ("frozen", "per_round"))
    if len(frozen) != 24 or len(corrected) != 208:
        raise ValueError("Unexpected match coverage")
    pd.testing.assert_frame_equal(frozen, refit, check_exact=True)
    return {
        "match_rows": len(corrected),
        "md1_twin_pairs": len(frozen),
        "max_exact_independent_vs_archived_grid": corrected.independent_grid_max_difference.max(),
    }


def compare_trial(adv, mapping):
    """Compare every team and target for each logged state to joint replicate zero."""
    mapping = mapping[["trial_state", "state_id"]].drop_duplicates()
    if mapping.trial_state.duplicated().any() or mapping.state_id.duplicated().any():
        raise ValueError("Logged state identities differ from the sensitivity experiment")
    by_trial = mapping.set_index("trial_state").state_id.to_dict()
    checked = set()
    with tarfile.open(TRIAL, "r|gz") as archive:
        for member in archive:
            if not (
                member.name.startswith("outputs/simulations/")
                and member.name.endswith("_joint_0.json")
            ):
                continue
            row = json.load(archive.extractfile(member))
            key = row["state_id"]
            archived = pd.DataFrame(row["advancement"]).set_index("team").sort_index()
            current = adv.loc[by_trial[key]].sort_index()
            pd.testing.assert_frame_equal(current, archived[current.columns], check_exact=True)
            checked.add(key)
    if checked != set(by_trial):
        raise ValueError("Missing original-seed trial result")
    return len(checked)


def check_outages(outages, rates, states):
    # Cross-check with the fitted model's own predict method and a fresh feature
    # batch for each arm. No replay prediction or feature cache participates.
    pins = pd.read_csv(PINS).query("model_name == @MODEL").set_index(["cadence_role", "regime"])
    fixtures = pd.read_csv("data/analysis/live/fixtures.csv")
    specs = {
        s.label: s
        for s in build_entropy_snapshot_specs(
            fixtures.assign(kickoff_utc=pd.to_datetime(fixtures.kickoff_utc, utc=True))
        )
    }
    names = {
        "r32_japan_brazil_locked": {"Japan", "Brazil"},
        "r32_germany_paraguay_locked": {"Germany", "Paraguay"},
        "r16_brazil_norway_locked": {"Brazil", "Norway"},
    }
    checks = []
    for r in outages.itertuples():
        spec = specs[r.snapshot_label]
        role = r.cadence_mode
        version = int(
            pins.loc[(role, "pre_tournament" if role == "frozen" else spec.round_seed), "version"]
        )
        if version != r.version:
            raise ValueError(f"Wrong outage model for {r.inference_run_id}")
        cutoff = SNAPSHOTS[r.snapshot_label]
        if pd.Timestamp(r.inference_timestamp) != cutoff or pd.Timestamp(r.cutoff) != cutoff:
            raise ValueError(f"Wrong timestamp for {r.inference_run_id}")
        wc = public_results(cutoff)
        gold = load_gold(BUNDLE / "gold" / r.gold_hash / "matches.parquet")
        gold, date = rc.augment_gold_for_inference(gold, wc)
        batch = build_inference_features(generate_all_wc_pairings(reference_date=date), gold)
        current = rates.get_group(r.state_id).set_index(SIDES)
        x = batch.set_index(SIDES).loc[current.index, FEATURES].to_numpy(dtype=float)
        state = json.loads((BUNDLE / f"models/v{version}/numeric_state.json").read_text())
        model = BivariatePoisson()
        model._scaler = StandardScaler()
        model._scaler.mean_ = np.array(state["mean_"])
        model._scaler.scale_ = np.array(state["scale_"])
        for i in (1, 2, 3):
            setattr(model, f"_coef{i}", np.array(state[f"_coef{i}"]))
        np.testing.assert_allclose(
            np.column_stack(model.predict(x)), current[MEANS], rtol=0, atol=TOL
        )
        common = np.exp(model._build_design(model._scaler.transform(x)) @ model._coef3).clip(1e-6)
        np.testing.assert_allclose(common, current[SHARED], rtol=0, atol=TOL)
        locks = states[r.state_id]["locks"]["ko"]
        label_pair = names.get(r.snapshot_label)
        if locks != sorted(wc["ko_results"].values(), key=lambda x: (x["home"], x["away"])):
            raise ValueError(f"Wrong result locks for {r.inference_run_id}")
        if label_pair and frozenset(label_pair) not in wc["ko_results"]:
            raise ValueError(f"Missing result for {r.snapshot_label}")
        checks.append(
            {
                "run": r.inference_run_id,
                "cutoff_utc": cutoff.isoformat(),
                "version": version,
                "settled_ko_matches": len(locks),
                "named_match_locked": (
                    None
                    if label_pair is None
                    else any({m["home"], m["away"]} == label_pair for m in locks)
                ),
                "max_archived_mean_difference": r.max_archived_mean_difference,
            }
        )
    return checks


def main():
    checks = check_matches()
    keys = ["inference_run_id", "cadence_mode", "model_name"]
    baseline = poiss_rows("rq2_entropy.csv", keys)
    cycles = pd.read_csv(OUT / "cycles.csv")
    pd.testing.assert_index_equal(
        baseline.set_index(keys).sort_index().index,
        cycles.set_index(keys, verify_integrity=True).sort_index().index,
    )
    states = json.loads((PREP / "states.json").read_text())
    rates = pd.read_parquet(PREP / "rates.parquet").groupby("state_id")
    adv = pd.read_parquet(OUT / "advancement.parquet").set_index(
        ["state_id", "team"], verify_integrity=True
    )
    old = pd.read_csv(BUNDLE / "trial_cycle_map.csv").rename(columns={"state_id": "trial_state"})
    logged = cycles.loc[~cycles.synthetic].merge(
        old[["run_id", "trial_state"]],
        left_on="inference_run_id",
        right_on="run_id",
        validate="one_to_one",
    )
    if len(logged) != 845 or cycles.synthetic.sum() != 8:
        raise ValueError("Incorrect Poisson simulation coverage")
    checks["logged_states_matching_all_trial_probabilities"] = compare_trial(adv, logged)
    checks["logged_cycles"] = len(logged)
    checks["max_logged_marginal_error"] = logged.max_archived_mean_difference.max()
    checks["synthetic_checks"] = check_outages(cycles[cycles.synthetic], rates, states)
    checks["distinct_simulation_inputs"] = len(states)
    # Reproduce representative historical advancement tables with the legacy sampler.
    with tarfile.open(TRIAL) as archive:
        pilot = pd.read_csv(BytesIO(archive.extractfile("outputs/pilot.csv").read()))
    controls = []
    for key in pilot.loc[pilot.distribution == "independent", "state_id"]:
        r = logged.loc[logged.trial_state == key].iloc[0]
        simulated = simulate_state(
            r.state_id, rates.get_group(r.state_id), states[r.state_id], joint=False
        )
        strand = "strand5_frozen_entropy" if r.cadence_mode == "frozen" else "strand2_brackets"
        ref = (
            BUNDLE
            / "advancement"
            / strand
            / r.cadence_mode
            / MODEL
            / f"{r.inference_run_id}_advancement.csv"
        )
        expected = pd.read_csv(ref).set_index("team").sort_index()
        actual = simulated.set_index("team").sort_index()
        np.testing.assert_array_equal(actual.index, expected.index)
        np.testing.assert_array_equal(
            actual[list(rc.ENTROPY_COLUMNS)], expected[list(rc.ENTROPY_COLUMNS)]
        )
        controls.append(r.inference_run_id)
    checks["exact_independent_controls"] = controls
    CHECKS.write_text(json.dumps(checks, indent=2) + "\n")
    print(
        f"Verified {len(logged)} logged cycles, eight synthetic forecasts and {checks['match_rows']} match rows"
    )


if __name__ == "__main__":
    main()
