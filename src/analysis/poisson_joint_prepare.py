"""Recover only Bivariate Poisson's historical score components for the thesis."""

from __future__ import annotations

import hashlib
import json
import tarfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import skellam

from src.analysis import replay_common as rc
from src.analysis.poisson_refit_diagnostic import _components
from src.analysis.rq2_outages import SNAPSHOTS, public_results
from src.analysis.strand4_entropy import build_entropy_snapshot_specs
from src.inference.features import build_inference_features, generate_all_wc_pairings
from src.models.config import MODEL_FEATURE_SETS
from src.models.data_split import load_gold
from src.models.evaluation import compute_rps

MODEL = "poisson_glm"
BUNDLE = Path("data/thesis_inputs/poisson")
BASE = Path("data/thesis_inputs/baseline/data/analysis")
PREP = Path("data/analysis/poisson_joint_inputs")
PINS = Path("data/reconstruction/inputs/models/manifest.csv")
LOGGED = Path("data/reconstruction/inputs/logged_predictions/all_models.parquet")
SIDES = ["home_team", "away_team"]
MEANS = ["lambda_h", "lambda_a"]
SHARED = "lambda_shared"
PROBS = ["p_home", "p_draw", "p_away"]
FEATURES = MODEL_FEATURE_SETS[MODEL]
TOL = 1e-12


def outcome_probs(unique_home, unique_away) -> np.ndarray:
    """Home win, draw, away win. The common goal count cancels from H-A."""
    h, a = np.broadcast_arrays(np.atleast_1d(unique_home), np.atleast_1d(unique_away))
    if not np.isfinite([h, a]).all() or (h < 0).any() or (a < 0).any():
        raise ValueError("Invalid unique goal rates")
    both = (h > 0) & (a > 0)
    p = np.zeros((h.size, 3))
    p[:, 1] = np.exp(-(h + a))
    p[:, 0] = np.where(a == 0, -np.expm1(-h), 0.0)
    p[:, 2] = np.where(h == 0, -np.expm1(-a), 0.0)
    p[both, 0] = skellam.sf(0, h[both], a[both])
    p[both, 1] = skellam.pmf(0, h[both], a[both])
    p[both, 2] = skellam.cdf(-1, h[both], a[both])
    return p


def fixture_rates(rates: pd.DataFrame, home: str, away: str, *, prefer_home: bool):
    forward = (rates.home_team == home) & (rates.away_team == away)
    reverse = (rates.home_team == away) & (rates.away_team == home)
    subset = rates.loc[forward if prefer_home and forward.any() else forward | reverse]
    if subset.empty:
        raise ValueError(f"No historical pairing for {home} vs {away}")
    row = subset.iloc[0]
    if row.home_team == home:
        return row.lambda_h, row.lambda_a, row.lambda_shared
    return row.lambda_a, row.lambda_h, row.lambda_shared


def gold_index() -> list[rc.GoldCommit]:
    with tarfile.open(BUNDLE / "gold_history.tar.gz") as archive:
        index = []
        for line in archive.extractfile("commits.txt").read().decode().splitlines():
            sha, timestamp = line.split()
            lock = yaml.safe_load(archive.extractfile(f"locks/{sha}.lock"))
            digest = lock["stages"]["gold"]["outs"][0]["md5"]
            index.append(rc.GoldCommit(sha, pd.Timestamp(timestamp), digest))
    return sorted(index, key=lambda commit: commit.commit_time)


class HistoricalRates:
    def __init__(self):
        self.index = gold_index()
        self.models = {}
        self.created = {}
        if json.loads((BUNDLE / "models/features.json").read_text()) != FEATURES:
            raise ValueError("Poisson feature order differs from the fitted models")
        for model in json.loads((BUNDLE / "models/manifest.json").read_text()):
            v = model["version"]
            if model["tags"]["model_name"] != MODEL:
                raise ValueError(f"Unexpected model in v{v}")
            path = BUNDLE / f"models/v{v}/numeric_state.json"
            self.models[v] = {k: np.array(x) for k, x in json.loads(path.read_text()).items()}
            self.created[v] = pd.Timestamp(model["created_utc"])
        self.feature_batches = {}

    @staticmethod
    @lru_cache(maxsize=4)
    def gold(digest):
        frame = load_gold(BUNDLE / "gold" / digest / "matches.parquet")
        # Only these historical columns enter the fitted feature vector. Include row order.
        cols = [
            "fixture_id",
            "date_utc",
            *SIDES,
            "home_goals",
            "away_goals",
            "home_elo_pre",
            "away_elo_pre",
            "home_elo_post",
            "away_elo_post",
        ]
        signature = hashlib.sha256(
            pd.util.hash_pandas_object(frame[cols], index=False).values
        ).digest()
        return frame, signature

    def commit_before(self, timestamp):
        commit = rc.resolve_gold_commit(timestamp, self.index)
        if commit is None:
            raise ValueError(f"No Gold snapshot before {timestamp}")
        return commit

    def predict(self, commit, wc, version, as_of, legacy_pairs=None):
        if self.created[version] > as_of or commit.commit_time >= as_of:
            raise ValueError(f"Future model or Gold for {as_of}, v{version}, {commit.commit_sha}")
        gold, signature = self.gold(commit.gold_hash)
        known = tuple(f["fixture_id"] for f in wc["finished_fixtures"])
        key = (signature, known, legacy_pairs)
        if key not in self.feature_batches:
            augmented, date = rc.augment_gold_for_inference(gold, wc)
            pairs = generate_all_wc_pairings(reference_date=date)
            if legacy_pairs is not None:
                # Early runs served one orientation per host-host pair. Fixture IDs
                # used the alphabetical pair even when a host was placed first.
                pairs = pairs.set_index(SIDES).loc[list(legacy_pairs)].reset_index()
                pairs["fixture_id"] = [
                    "wc2026_pair_" + "_".join(sorted(pair)) for pair in legacy_pairs
                ]
            features = build_inference_features(pairs, augmented)
            self.feature_batches[key] = features.set_index(SIDES)[FEATURES]
        features = self.feature_batches[key]
        h, a, common = _components(self.models[version], features.to_numpy(dtype=float))
        rates = features.index.to_frame(index=False)
        rates["lambda_h"], rates["lambda_a"], rates[SHARED] = h + common, a + common, common
        return rates


def pin_rates(history, commit, wc, version, timestamp, original):
    keys = pd.MultiIndex.from_frame(original[SIDES])
    target = original[MEANS].to_numpy()

    def aligned(v, legacy=None):
        rates = history.predict(commit, wc, v, timestamp, legacy)
        return rates.set_index(SIDES).loc[keys].reset_index()

    rates = aligned(version)
    convention = "full"
    if len(original) == 1128 and not np.allclose(rates[MEANS], target, rtol=0, atol=TOL):
        rates = aligned(version, tuple(keys))
        convention = "early_host_ids"
    error = float(np.abs(rates[MEANS].to_numpy() - target).max())
    if not np.isfinite(rates[[*MEANS, SHARED]].to_numpy()).all():
        raise ValueError(f"Nonfinite goal rates at {timestamp}, v{version}")
    np.testing.assert_allclose(rates[MEANS], target, rtol=0, atol=TOL,
                               err_msg=f"v{version}, {timestamp}, {commit.gold_hash}")
    rates[MEANS] = target  # retain the exact archived floating-point representation
    return rates, convention, error


def read_by_run(path):
    rows = pd.read_parquet(path, filters=[("model_name", "==", MODEL)])
    return {run: frame for run, frame in rows.groupby("inference_run_id", sort=False)}


def poiss_rows(filename, keys):
    rows = pd.read_csv(BASE / filename, dtype={"round_label": str, "matchday_label": str})
    rows = rows.loc[rows.model_name == MODEL].copy()
    if rows.empty or rows.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate Poisson keys in {filename}")
    return rows


def match_source(cadence: str, round_label: str, provenance: str) -> str:
    if cadence == "frozen" and provenance == "reconstructed_frozen_shadow":
        return "frozen"
    if cadence == "per_round":
        if round_label == "1" and provenance == "md1_frozen_copied":
            return "frozen"
        if round_label != "1" and provenance == "live":
            return "live"
    raise ValueError(f"Unexpected match provenance: {cadence}, {round_label}, {provenance}")


def prepare_cycles(history, matches, entropy):
    live = rc.load_inference_cycles().set_index(
        "inference_run_id", drop=False, verify_integrity=True
    )
    trial = pd.read_csv(BUNDLE / "trial_cycle_map.csv").set_index("run_id", verify_integrity=True)
    if set(entropy.loc[~entropy.synthetic, "inference_run_id"]) != set(trial.index):
        raise ValueError("Archived model pins do not cover the exact logged cycles")
    snapshots = {
        s.label: s for s in build_entropy_snapshot_specs(matches.drop_duplicates("match_id"))
    }
    pins = pd.read_csv(PINS).query("model_name == @MODEL").set_index(["cadence_role", "regime"])
    frozen_version = int(pins.loc[("frozen", "pre_tournament"), "version"])
    frozen = read_by_run(BUNDLE / "strand5_predictions.parquet")
    logged = read_by_run(LOGGED)
    states, rate_tables, cycle_rows, match_sources = {}, [], [], {}
    live_matches = {
        run: group
        for run, group in matches.loc[matches.provenance == "live"].groupby("inference_run_id")
    }

    for row in entropy.itertuples():
        run = row.inference_run_id
        timestamp = pd.Timestamp(row.inference_timestamp)
        commit = history.commit_before(timestamp)
        if row.synthetic:
            spec = snapshots[row.snapshot_label]
            if timestamp != spec.synthetic_timestamp:
                raise ValueError(f"Invalid synthetic timestamp for {run}")
            version = (
                frozen_version
                if row.cadence_mode == "frozen"
                else int(pins.loc[("per_round", spec.round_seed), "version"])
            )
            original = frozen[run]
            # Both archived synthetic arms served the frozen matrix at the old cutoff.
            pin_rates(history, commit, rc.parse_wc_results_before_kickoff(spec.max_kickoff),
                      frozen_version, timestamp, original)
            timestamp = cutoff = SNAPSHOTS[row.snapshot_label]
            commit = history.commit_before(timestamp)
            wc = public_results(cutoff)
            rates = history.predict(commit, wc, version, timestamp)
            rates = rates.set_index(SIDES).loc[pd.MultiIndex.from_frame(original[SIDES])].reset_index()
            error = float(np.abs(rates[MEANS].to_numpy() - original[MEANS].to_numpy()).max())
            batch = "full"
            seed, n = rc._seed_from_string(spec.round_seed), 10_000
        else:
            cycle = rc._cycle_from_row(live.loc[run])
            if (
                cycle.inference_timestamp != timestamp
                or cycle.cadence_mode != row.cadence_mode
                or cycle.simulation_seed is None
                or cycle.n_sims != 10_000
            ):
                raise ValueError(f"Invalid cycle parameters for {run}")
            pin = trial.loc[run]
            version = int(pin.version)
            if pin.gold_hash != commit.gold_hash or (
                row.cadence_mode == "frozen" and version != frozen_version
            ):
                raise ValueError(f"Incorrect historical pin for {run}")
            cutoff = timestamp
            wc = rc.parse_wc_results_before_kickoff(cutoff, settle_delta=rc.SETTLE_DELTA)
            seed, n = cycle.simulation_seed, cycle.n_sims
            source = frozen if row.cadence_mode == "frozen" else logged
            rates, batch, error = pin_rates(history, commit, wc, version, timestamp, source[run])
        locks = {
            "group": [[h, a, *score] for (h, a), score in sorted(wc["group_results"].items())],
            "ko": sorted(wc["ko_results"].values(), key=lambda x: (x["home"], x["away"])),
        }
        parameters = {"locks": locks, "seed": seed, "n_sims": n}
        digest = hashlib.sha256(pd.util.hash_pandas_object(rates, index=False).values)
        digest.update(json.dumps(parameters, sort_keys=True).encode())
        state = digest.hexdigest()
        if state not in states:
            states[state] = parameters
            rate_tables.append(rates.assign(state_id=state))
        provenance = {
            "version": version,
            "gold_hash": commit.gold_hash,
            "gold_commit": commit.commit_sha,
            "cutoff": str(cutoff),
            "batch": batch,
            "known_fixture_ids": ",".join(str(f["fixture_id"]) for f in wc["finished_fixtures"]),
        }
        cycle_rows.append(
            dict(
                inference_run_id=run,
                inference_timestamp=timestamp,
                model_name=MODEL,
                cadence_mode=row.cadence_mode,
                state_id=state,
                synthetic=row.synthetic,
                snapshot_label=row.snapshot_label,
                max_archived_mean_difference=error,
                **provenance,
            )
        )
        if run in live_matches:
            for match in live_matches[run].itertuples():
                if row.cadence_mode != "per_round" or timestamp >= match.kickoff_utc:
                    raise ValueError(f"Invalid live forecast source for match {match.match_id}")
                match_sources[(match.match_id, "live")] = (
                    fixture_rates(rates, match.home, match.away, prefer_home=True),
                    provenance,
                )
    return states, rate_tables, cycle_rows, match_sources, frozen_version


def prepare_matches(history, matches, match_sources, frozen_version):
    for match in matches.loc[matches.cadence_mode == "frozen"].itertuples():
        commit = history.commit_before(match.kickoff_utc)
        wc = rc.parse_wc_results_before_kickoff(match.kickoff_utc, settle_delta=rc.SETTLE_DELTA)
        known = [str(f["fixture_id"]) for f in wc["finished_fixtures"]]
        if str(match.match_id) in known:
            raise ValueError(f"Match {match.match_id} present in its own forecast history")
        rates = history.predict(commit, wc, frozen_version, match.kickoff_utc)
        match_sources[(match.match_id, "frozen")] = (
            fixture_rates(rates, match.home, match.away, prefer_home=False),
            {
                "version": frozen_version,
                "gold_hash": commit.gold_hash,
                "gold_commit": commit.commit_sha,
                "cutoff": str(match.kickoff_utc),
                "batch": "full",
                "known_fixture_ids": ",".join(known),
            },
        )
    records = []
    for m in matches.itertuples():
        source = match_source(m.cadence_mode, m.round_label, m.provenance)
        (h, a, shared), provenance = match_sources[(m.match_id, source)]
        np.testing.assert_allclose(
            [h, a],
            [m.lambda_h, m.lambda_a],
            atol=TOL,
            rtol=0,
            err_msg=f"Mismatch for {m.match_id} {m.cadence_mode}",
        )
        p = outcome_probs(m.lambda_h - shared, m.lambda_a - shared)
        exact_independent = outcome_probs(m.lambda_h, m.lambda_a)[0]
        grid_delta = np.abs(exact_independent - [m.p_home, m.p_draw, m.p_away]).max()
        records.append(
            dict(
                match_id=m.match_id,
                model_name=MODEL,
                cadence_mode=m.cadence_mode,
                inference_run_id=m.inference_run_id,
                source_run_id=m.source_run_id,
                provenance=m.provenance,
                lambda_h=m.lambda_h,
                lambda_a=m.lambda_a,
                lambda_shared=shared,
                **dict(zip(PROBS, p[0])),
                rps=compute_rps(p, np.array([m.actual_outcome]))[0],
                independent_grid_max_difference=grid_delta,
                **provenance,
            )
        )
    return pd.DataFrame(records)


def prepare():
    history = HistoricalRates()
    matches = poiss_rows("rq1_matches.csv", ["match_id", "cadence_mode"])
    matches["kickoff_utc"] = pd.to_datetime(matches.kickoff_utc, utc=True)
    entropy = poiss_rows("rq2_entropy.csv", ["inference_run_id", "cadence_mode"])
    states, rate_tables, cycle_rows, match_sources, frozen_version = prepare_cycles(
        history, matches, entropy
    )
    records = prepare_matches(history, matches, match_sources, frozen_version)
    if (len(records), len(cycle_rows), sum(r.synthetic for r in entropy.itertuples())) != (
        208,
        853,
        8,
    ):
        raise ValueError("Incomplete historical Poisson coverage")
    PREP.mkdir(parents=True, exist_ok=True)
    records.to_csv(PREP / "matches.csv", index=False)
    pd.DataFrame(cycle_rows).to_csv(PREP / "cycles.csv", index=False)
    pd.concat(rate_tables, ignore_index=True).to_parquet(PREP / "rates.parquet", index=False)
    (PREP / "states.json").write_text(json.dumps(states, sort_keys=True))
    print(
        f"Verified {len(records)} match and {len(cycle_rows)} cycle sources, {len(states)} simulation inputs"
    )


if __name__ == "__main__":
    prepare()
