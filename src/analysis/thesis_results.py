"""Verify the official joint-Poisson analysis against the archived full roster."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.analysis.thesis_datasets import (
    BASE,
    CYCLE_KEYS,
    JOINT,
    MATCH_KEYS,
    MODEL,
    OUT,
    OUTAGES,
    PROBS,
    TARGETS,
    read_fields,
)
from src.models.evaluation import compute_rps

ROOT = Path("data/analysis")
REG = ROOT / "regulation_time"
TABLES = Path("docs/tables")
ARCHIVE = Path("data/thesis_inputs/baseline")
PAIRS = ROOT / "rq2_paired_diffs.csv"
FIG = Path("docs/figures")
REPORT = ROOT / "thesis_results.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(left, right):
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def indexed(path, keys):
    return read_fields(path).set_index(keys, verify_integrity=True).sort_index()


def check_sources():
    original = indexed(BASE / "rq1_matches.csv", MATCH_KEYS)
    matches = indexed(OUT / "rq1_matches.csv", MATCH_KEYS)
    other = matches.index.get_level_values("model_name") != MODEL
    changed = [*PROBS, "rps"]
    same(original[other], matches[other])
    same(original.drop(columns=changed), matches.drop(columns=changed))
    same(matches.loc[~other, changed], indexed(JOINT / "matches.csv", MATCH_KEYS)[changed])
    p = matches.loc[~other, PROBS].astype(float).to_numpy()
    np.testing.assert_allclose(
        compute_rps(p, matches.loc[~other, "actual_outcome"].astype(int)),
        matches.loc[~other, "rps"].astype(float),
        atol=1e-12,
        rtol=0,
    )
    cycles = indexed(OUT / "rq2_entropy.csv", CYCLE_KEYS)
    baseline = indexed(BASE / "rq2_entropy.csv", CYCLE_KEYS)
    synth = cycles.synthetic == "True"
    poisson = cycles.index.get_level_values("model_name") == MODEL
    fields = ["inference_timestamp", *TARGETS]
    same(cycles.drop(columns=fields), baseline.drop(columns=fields))
    same(cycles.loc[~synth, ["inference_timestamp"]], baseline.loc[~synth, ["inference_timestamp"]])
    same(cycles[~synth & ~poisson], baseline[~synth & ~poisson])
    replay = indexed(JOINT / "cycles.csv", CYCLE_KEYS)
    same(cycles.loc[poisson, TARGETS], replay[TARGETS])
    same(cycles.loc[synth, fields], indexed(OUTAGES, CYCLE_KEYS)[fields])
    counts = (len(matches), len(matches[~other]), len(cycles), int(poisson.sum()), int(synth.sum()))
    require(counts == (1664, 208, 3412, 853, 32), f"Unexpected source coverage {counts}")
    return {
        "match_rows": counts[0],
        "poisson_match_rows": counts[1],
        "cycle_rows": counts[2],
        "poisson_cycle_rows": counts[3],
        "synthetic_cycle_rows": counts[4],
        "poisson_match_fields": changed,
        "poisson_logged_cycle_fields": TARGETS,
        "all_synthetic_cycle_fields": fields,
    }


def check_regulation():
    m = indexed(REG / "rq1_matches.csv", MATCH_KEYS)
    old = indexed(BASE / "regulation_time/rq1_matches.csv", MATCH_KEYS)
    other = m.index.get_level_values("model_name") != MODEL
    same(m[other], old[other])
    same(m.drop(columns=[*PROBS, "rps"]), old.drop(columns=[*PROBS, "rps"]))
    twins = m[~other & (m.round_label == "1")][[*PROBS, "rps"]]
    same(twins.xs("frozen", level="cadence_mode"), twins.xs("per_round", level="cadence_mode"))
    arms = m[~other].reset_index().groupby("cadence_mode").size().to_dict()
    require(arms == {"frozen": 104, "per_round": 104} and len(twins) == 48, "Wrong Poisson sample")
    bins = pd.read_csv(REG / "rq3_calibration.csv").query("record_type == 'reliability'")
    require(set(bins.outcome_class) == {"home", "draw", "away"}, "Missing calibration class")
    treated = m[~other & (m.round_label != "1")].reset_index().groupby("cadence_mode").size()
    require(treated.eq(80).all(), "Wrong treated match coverage")
    for _, rows in bins.groupby("outcome_class"):
        actual = rows.groupby(["cadence_mode", "model_name", "round_label"]).n.sum()
        expected = m.reset_index().groupby(actual.index.names).size()
        pd.testing.assert_series_equal(actual, expected, check_names=False)
    return {"matches_per_arm": arms, "md1_pairs": len(twins) // 2, "treated_matches": 80}


def check_pairs():
    keys = ["model_name", "run_id_per_round", "run_id_frozen"]
    pairs = pd.read_csv(PAIRS).set_index(keys, verify_integrity=True).sort_index()
    old = pd.read_csv(BASE / PAIRS.name).set_index(keys, verify_integrity=True).sort_index()
    pd.testing.assert_index_equal(pairs.index, old.index)
    other = pairs.index.get_level_values("model_name") != MODEL
    same(pairs[other], old[other])
    synth = pd.read_csv(OUTAGES).inference_run_id
    for col in keys[1:]:
        require(
            not pairs.index.get_level_values(col).isin(synth).any(), "Synthetic pair in inference"
        )
    new, prior = [
        p.xs(MODEL).loc[lambda x: ~x.matchday_label.isin(["1", "2"]) & x.is_new_state]
        for p in (pairs, old)
    ]
    require(len(prior) == 88, "Incorrect archived RQ2 sample")
    fixed = pairs.xs(MODEL).loc[prior.index]
    return {
        "archived_states": len(prior),
        "official_states": len(new),
        "added_pairs": list(new.index.difference(prior.index)),
        "removed_pairs": list(prior.index.difference(new.index)),
        "corrected_means_on_archived_pairs": fixed[["d_" + t for t in TARGETS]].mean().to_dict(),
    }


def check_tables():
    files = sorted((TABLES / "regulation_time").glob("*.csv")) + sorted(
        TABLES.glob("rq2_paired_*.csv")
    )
    adjusted_changes = []
    for path in files:
        rows = pd.read_csv(path)
        old = pd.read_csv(ARCHIVE / path)
        if "p_holm" in rows and rows.p_holm.notna().any():
            keys = (
                ["window", "grain", "model_name", "stage"]
                if "stage" in rows
                else ["phase_scheme", "phase", "model_name", "metric"]
            )
            new = rows.dropna(subset="p_holm").set_index(keys, verify_integrity=True).p_holm
            previous = old.dropna(subset="p_holm").set_index(keys, verify_integrity=True).p_holm
            pd.testing.assert_index_equal(new.index, previous.index)
            for key in new.index[new.ne(previous)]:
                adjusted_changes.append(
                    {**dict(zip(keys, key)), "old": previous[key], "new": new[key]}
                )
        if "model_name" in rows:
            same(
                rows.loc[rows.model_name != MODEL]
                .drop(columns="p_holm", errors="ignore")
                .reset_index(drop=True),
                old.loc[old.model_name != MODEL]
                .drop(columns="p_holm", errors="ignore")
                .reset_index(drop=True),
            )
    accuracy = pd.read_csv(TABLES / "regulation_time/rq1_paired_phase.csv").dropna(subset="p_holm")
    entropy = pd.read_csv(TABLES / "rq2_paired_stage.csv").dropna(subset="p_holm")
    calibration = pd.read_csv(TABLES / "regulation_time/rq3_calibration_phase.csv").dropna(
        subset="p_holm"
    )
    families = {
        "rps": accuracy.query("metric == 'rps'"),
        "nll": accuracy.query("metric == 'nll'"),
        "entropy": entropy,
        "calibration": calibration,
    }
    counts = {name: len(f) for name, f in families.items()}
    require(list(counts.values()) == [9, 9, 18, 18], f"Incorrect Holm families {counts}")
    for f in families.values():
        # The published CSVs round values to six significant digits; the estimators
        # apply Holm before rounding.
        np.testing.assert_allclose(
            multipletests(f.p_perm, method="holm")[1], f.p_holm, atol=1e-5, rtol=0
        )
    boundary = pd.read_csv(TABLES / "rq2_refit_boundary.csv")
    require(
        boundary.groupby("baseline").target.nunique().to_dict()
        == {"live_per_round": 6, "plotted": 6},
        "Missing refit boundary",
    )
    return counts, adjusted_changes


def main():
    source = ARCHIVE / "source.json"
    identities = json.loads(source.read_text())
    for relative, digest in identities["files"].items():
        require(sha(ARCHIVE / relative) == digest, f"Archived input changed: {relative}")
    products = [
        *OUT.glob("*.csv"),
        *REG.glob("*.csv"),
        PAIRS,
        ROOT / "round_leaderboards.csv",
        *sorted((TABLES / "regulation_time").glob("*.csv")),
        *TABLES.glob("rq2_*.csv"),
        FIG / "entropy-trajectory.png",
        FIG / "entropy-stages.png",
        *(FIG / "entropy-targets").glob("*.png"),
        *(FIG / "regulation-time").glob("*.png"),
        ROOT / "poisson_joint_checks.json",
        OUTAGES,
    ]
    require(sum(p.suffix == ".png" for p in products) == 10, "Incomplete thesis figures")
    families, adjusted_changes = check_tables()
    report = {
        "archived_revision": identities["revision"],
        "archive_manifest_sha256": sha(source),
        "input_sha256": {
            str(p): sha(p)
            for p in [
                Path("data/thesis_inputs/poisson/models/manifest.json"),
                Path("data/thesis_inputs/sensitivity.tar.gz"),
            ]
        },
        "sources": check_sources(),
        "regulation_time": check_regulation(),
        "rq2_treated_poisson": check_pairs(),
        "holm_family_sizes": families,
        "changed_adjusted_p_values": adjusted_changes,
        "sha256": {str(p): sha(p) for p in sorted(products)},
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Verified archived inputs, replacements, comparisons and {len(products)} outputs")


if __name__ == "__main__":
    main()
