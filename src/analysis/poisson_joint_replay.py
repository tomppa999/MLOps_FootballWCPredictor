"""Simulate each distinct historical Poisson tournament input at its original seed."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.replay_common import ENTROPY_COLUMNS, compute_entropy_columns
from src.inference.simulation import simulate_tournament

PREP = Path("data/analysis/poisson_joint_inputs")
OUT = Path("data/analysis/poisson_joint")


def simulate_state(state_id, rates, params, *, joint=True):
    group = {(h, a): (hg, ag) for h, a, hg, ag in params["locks"]["group"]}
    ko = {frozenset((r["home"], r["away"])): r for r in params["locks"]["ko"]}
    result = simulate_tournament(
        rates, n_sims=params["n_sims"], seed=params["seed"], joint_poisson=joint,
        locked_group_results=group, locked_ko_results=ko)
    adv = result["advancement"].sort_values("team").reset_index(drop=True)
    adv = adv[["team", *ENTROPY_COLUMNS]].copy()
    values = adv[list(ENTROPY_COLUMNS)].to_numpy()
    if len(adv) != 48 or adv.team.duplicated().any() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"Invalid advancement probabilities for {state_id}")
    np.testing.assert_allclose(values.sum(axis=0), [32, 16, 8, 4, 2, 1], atol=1e-10, rtol=0)
    return adv.assign(state_id=state_id)


def main():
    params = json.loads((PREP / "states.json").read_text())
    rates = pd.read_parquet(PREP / "rates.parquet")
    cycles = pd.read_csv(PREP / "cycles.csv")
    frames = rates.groupby("state_id", sort=False)
    if set(frames.groups) != set(params) or set(cycles.state_id) != set(params):
        raise ValueError("Simulation state keys disagree")
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(simulate_state, key, rows, params[key]) for key, rows in frames]
        results = []
        for i, task in enumerate(tasks, 1):
            results.append(task.result())
            if i % 50 == 0:
                print(f"Simulated {i}/{len(tasks)} distinct tournament inputs", flush=True)
    advancement = pd.concat(results, ignore_index=True)
    entropies = pd.DataFrame([dict(state_id=key, **compute_entropy_columns(rows))
                              for key, rows in advancement.groupby("state_id", sort=False)])
    cycles = cycles.merge(entropies, on="state_id", validate="many_to_one")
    OUT.mkdir(parents=True, exist_ok=True)
    advancement.to_parquet(OUT / "advancement.parquet", index=False)
    cycles.to_csv(OUT / "cycles.csv", index=False)
    # Preserve the full-precision match values and source IDs written by preparation.
    (OUT / "matches.csv").write_bytes((PREP / "matches.csv").read_bytes())
    print(f"Wrote {len(cycles)} corrected Poisson cycle forecasts from {len(tasks)} states")


if __name__ == "__main__":
    main()
