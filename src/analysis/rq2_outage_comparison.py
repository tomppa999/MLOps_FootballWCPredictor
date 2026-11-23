"""Compare four models on the same four publicly known tournament states."""

import json
import pickletools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import Booster

from src.analysis import replay_common as rc
from src.analysis.poisson_joint_prepare import BUNDLE, LOGGED, MEANS, MODEL, PINS, PREP, SIDES
from src.analysis.poisson_joint_replay import OUT, simulate_state
from src.analysis.rq2_outages import SNAPSHOTS, public_results
from src.inference.features import build_inference_features, generate_all_wc_pairings
from src.models.config import EXPERIMENT_MODELS, MODEL_FEATURE_SETS
from src.models.data_split import load_gold

OUTPUT = Path("data/analysis/rq2_public_outages.csv")


def check_advancement(frame, locks):
    by_team = frame.set_index("team")
    for match in locks:
        column = {"R32": "p_r16", "R16": "p_qf"}[match["stage"]]
        winner = match["winner"]
        loser = match["away"] if match["home"] == winner else match["home"]
        if by_team.at[winner, column] != 1 or by_team.at[loser, column] != 0:
            raise ValueError(f"Unresolved {match['stage']} result for {winner} vs {loser}")


@lru_cache(maxsize=9)
def numeric_parameters(path, name):
    """Read only literal numeric fields and native XGBoost buffers from pinned inputs."""
    fields = {}
    buffers = []
    field = None
    for op, arg, _ in pickletools.genops(path.read_bytes()):
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE"}:
            if arg in {
                "mean_",
                "var_",
                "scale_",
                "_beta_h",
                "_beta_a",
                "_lambda",
                "_intercept_h",
                "_intercept_a",
                "_beta_h_samples",
            }:
                field = arg
        elif op.name == "BINFLOAT" and field in {"_lambda", "_intercept_h", "_intercept_a"}:
            fields[field] = arg
            field = None
        elif op.name == "BYTEARRAY8":
            if name == "xgboost":
                # Each handle is the native fitted booster (home, away).
                booster = Booster.__new__(Booster)
                booster.__setstate__({"handle": arg})
                buffers.append(booster)
            elif field in {"mean_", "scale_", "_beta_h", "_beta_a"}:
                fields[field] = np.frombuffer(arg, dtype="<f8")
            field = None
    if name == "xgboost" and len(buffers) != 2:
        raise ValueError(f"Expected two fitted boosters in {path}")
    return fields, buffers


def rates_for(features, pin):
    name = pin.model_name
    x = features[MODEL_FEATURE_SETS[name]].to_numpy(dtype=float)
    path = rc.snapshot_model_dir(pin.registry_name, pin.version) / "python_model.pkl"
    fields, boosters = numeric_parameters(path, name)
    if name == "xgboost":
        h, a = (booster.inplace_predict(x) for booster in boosters)
    elif name == "mean_rate_poisson":
        h = a = np.full(len(x), fields["_lambda"])
    elif name == "bayesian_poisson":
        scaled = (x - fields["mean_"]) / fields["scale_"]
        h = np.exp(fields["_intercept_h"] + scaled @ fields["_beta_h"])
        a = np.exp(fields["_intercept_a"] + scaled @ fields["_beta_a"])
    else:
        raise ValueError(f"Unexpected comparison model {name}")
    return features[SIDES].assign(lambda_h=np.maximum(h, 1e-6), lambda_a=np.maximum(a, 1e-6))


def feature_batch(gold_hash, wc):
    gold = load_gold(BUNDLE / "gold" / gold_hash / "matches.parquet")
    gold, date = rc.augment_gold_for_inference(gold, wc)
    return build_inference_features(generate_all_wc_pairings(reference_date=date), gold)


def verify_control(row, pins):
    """Reproduce all marginal rates for the last observed cycle before the gap."""
    ts = pd.Timestamp(row.inference_timestamp)
    features = feature_batch(
        row.gold_hash, rc.parse_wc_results_before_kickoff(ts, settle_delta=rc.SETTLE_DELTA)
    )
    path = BUNDLE / "strand5_predictions.parquet" if row.cadence_mode == "frozen" else LOGGED
    archived = pd.read_parquet(path, filters=[("inference_run_id", "==", row.inference_run_id)])
    for pin in pins.itertuples():
        orig = archived.loc[archived.model_name == pin.model_name].set_index(SIDES).sort_index()
        actual = rates_for(features, pin).set_index(SIDES).sort_index()
        pd.testing.assert_index_equal(actual.index, orig.index)
        np.testing.assert_allclose(
            actual[MEANS],
            orig[MEANS],
            atol=1e-12,
            rtol=0,
            err_msg=f"{row.inference_run_id} {pin.model_name} v{pin.version}",
        )


def main():
    cycles = pd.read_csv(OUT / "cycles.csv")
    cycles["inference_timestamp"] = pd.to_datetime(
        cycles.inference_timestamp, format="ISO8601", utc=True
    )
    states = json.loads((PREP / "states.json").read_text())
    adv = pd.read_parquet(OUT / "advancement.parquet").groupby("state_id")
    pins = pd.read_csv(PINS)
    outputs = []
    verified = set()
    for row in cycles.loc[cycles.synthetic].itertuples():
        label, arm = row.snapshot_label, row.cadence_mode
        cutoff = SNAPSHOTS[label]
        wc = public_results(cutoff)
        features = feature_batch(row.gold_hash, wc)
        params = states[row.state_id]
        locks = params["locks"]["ko"]
        expected_locks = sorted(wc["ko_results"].values(), key=lambda m: (m["home"], m["away"]))
        if locks != expected_locks or row.inference_timestamp != cutoff:
            raise ValueError(f"Public results disagree with the Poisson replay for {label}")
        regime = "pre_tournament" if arm == "frozen" else label[:3].upper()
        others = pins.loc[
            (pins.regime == regime) & (pins.cadence_role == arm) & (pins.model_name != MODEL)
        ]
        previous = (
            cycles.loc[
                (~cycles.synthetic)
                & (cycles.cadence_mode == arm)
                & (cycles.inference_timestamp < cutoff)
            ]
            .sort_values("inference_timestamp")
            .iloc[-1]
        )
        if previous.inference_run_id not in verified:
            verify_control(previous, others)
            verified.add(previous.inference_run_id)
        forecasts = {MODEL: (row.version, adv.get_group(row.state_id))}
        for pin in others.itertuples():
            rates = rates_for(features, pin)
            result = simulate_state(label, rates, params, joint=False)
            forecasts[pin.model_name] = (pin.version, result)
        if set(forecasts) != set(EXPERIMENT_MODELS):
            raise ValueError(f"Incomplete model roster for {row.inference_run_id}")
        for name, (version, frame) in forecasts.items():
            check_advancement(frame, locks)
            outputs.append(
                dict(
                    inference_run_id=row.inference_run_id,
                    inference_timestamp=cutoff,
                    cadence_mode=arm,
                    model_name=name,
                    version=version,
                    synthetic=True,
                    snapshot_label=label,
                    known_fixture_ids=row.known_fixture_ids,
                    seed=params["seed"],
                    n_sims=params["n_sims"],
                    **rc.compute_entropy_columns(frame),
                )
            )
    result = pd.DataFrame(outputs)
    expected = pd.MultiIndex.from_product([SNAPSHOTS, ("frozen", "per_round"), EXPERIMENT_MODELS])
    keys = result.set_index(
        ["snapshot_label", "cadence_mode", "model_name"], verify_integrity=True
    ).index
    if set(keys) != set(expected):
        raise ValueError("Incomplete outage comparison")
    result.to_csv(OUTPUT, index=False)
    print(f"Verified {len(result)} outage forecasts and {len(verified)} pre-outage controls")


if __name__ == "__main__":
    main()
