"""Assemble full-roster thesis inputs, preserving unmodified archived CSV fields."""

from pathlib import Path

import pandas as pd

BASE = Path("data/thesis_inputs/baseline/data/analysis")
JOINT = Path("data/analysis/poisson_joint")
OUTAGES = Path("data/analysis/rq2_public_outages.csv")
OUT = Path("data/analysis/thesis")
MATCH_KEYS = ["match_id", "model_name", "cadence_mode"]
CYCLE_KEYS = ["inference_run_id", "model_name", "cadence_mode"]
PROBS = ["p_home", "p_draw", "p_away"]
TARGETS = [f"entropy_{stage}" for stage in ("r32", "r16", "qf", "sf", "final", "winner")]
MODEL = "poisson_glm"


def read_fields(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def replace_rows(base, mask, replacements, keys, fields):
    selected = base.loc[mask].set_index(keys, verify_integrity=True)
    source = replacements.set_index(keys, verify_integrity=True)
    if not selected.index.sort_values().equals(source.index.sort_values()):
        raise ValueError(
            f"Replacement keys differ: {selected.index.symmetric_difference(source.index).tolist()}"
        )
    aligned = source.loc[selected.index, fields]
    if aligned.isna().any().any() or aligned.eq("").any().any():
        raise ValueError("Incomplete replacement field")
    base.loc[mask, fields] = aligned.to_numpy()


def main():
    matches = read_fields(BASE / "rq1_matches.csv")
    cycles = read_fields(BASE / "rq2_entropy.csv")
    replace_rows(
        matches,
        matches.model_name == MODEL,
        read_fields(JOINT / "matches.csv"),
        MATCH_KEYS,
        [*PROBS, "rps"],
    )
    logged = read_fields(JOINT / "cycles.csv").query("synthetic == 'False'")
    synthetic = cycles.synthetic == "True"
    replace_rows(cycles, ~synthetic & (cycles.model_name == MODEL), logged, CYCLE_KEYS, TARGETS)
    replace_rows(
        cycles, synthetic, read_fields(OUTAGES), CYCLE_KEYS, ["inference_timestamp", *TARGETS]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUT / "rq1_matches.csv", index=False)
    cycles.to_csv(OUT / "rq2_entropy.csv", index=False)


if __name__ == "__main__":
    main()
