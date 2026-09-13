# WC 2026 Match Predictor — An End-to-End MLOps Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit App](https://img.shields.io/badge/streamlit-live-brightgreen)](https://wc2026-predictions.streamlit.app/)
[![MLflow on DagsHub](https://img.shields.io/badge/mlflow-dagshub-orange)](https://dagshub.com/tomppa999/MLOps_FootballWCPredictor)

An end-to-end MLOps pipeline for predicting FIFA World Cup 2026 match outcomes. The system ingests football national team match data every 2 hours, rebuilds features, refits models, and simulates the full 48-team tournament bracket — all running automatically on GCP Cloud Run. Built alongside a thesis on the effect of regular retraining and uncertainty resolution in tournament prediction.

**Live dashboard:** [wc2026-predictions.streamlit.app](https://wc2026-predictions.streamlit.app/)

---

## What's running right now

- **Automated pipeline** on GCP Cloud Run Jobs, scheduled daily pre-tournament and every 2 hours during WC 2026 (June 11 – July 19)
- **Live data ingestion** from API-Football + Elo ratings; ~6,900 match rows in the Gold dataset
- **4-model production roster:** `mean_rate_poisson` (baseline), `poisson_glm`, `bayesian_poisson`, `xgboost`
- **Cadence experiment:** two champion variants run in parallel — a model frozen before the tournament starts vs. a model refitted after each completed round. The experiment measures whether per-round refits meaningfully resolve prediction uncertainty as the tournament progresses.
- **Live monitoring:** RPS and NLL scored against all roster models after each settled match
- **Streamlit dashboard:** group standings, match predictions, knockout path probabilities

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   pipeline/trigger.py                    │
│       (Cloud Run Job, every 2h during WC 2026)           │
└──────┬──────────────────────────────────────┬───────────┘
       │                                      │
       ▼                                      ▼
ingestion/                              inference/
API-Football + Elo TSVs                 predict → simulate
       │                                      │
       ▼                                      ▼
  Bronze (raw)                       Monte Carlo bracket sim
       │                                      │
       ▼                                      ▼
  Silver (clean)                     dashboard/ (Streamlit)
       │
       ▼
  Gold (features, 1 row/match)
       │
       ▼
  models/ (train → tune → QA → promote)
       │
       ▼
  MLflow registry (DagsHub remote)
  DVC artifact versioning
```

Data flows Bronze → Silver → Gold via `dvc repro`. The trigger orchestrates everything: freshness checks, DVC pipeline, champion/shadow training, inference, simulation, and monitoring in a single daily (pre-WC) / every-2h (WC) cycle.

---

## Modeling approach

The prediction target is **goals scored per team per match** (a Poisson-family regression problem), not direct win/draw/loss classification. Match outcomes and tournament advancement probabilities are derived downstream via Monte Carlo simulation over the full bracket.

Eleven candidate models were evaluated (Poisson GLM, Negative Binomial GLM, XGBoost, Bayesian Poisson, Ridge, Random Forest, SARIMAX, LSTM, CNN, mean-rate Poisson baseline, and a bivariate variant). All experiments are logged to MLflow. Four models were promoted to the production roster based on holdout RPS.

Features are strictly time-aware and leakage-safe: all rolling stats, Elo ratings, and competition context use only information available before each match.

### Results (expanded holdout, ~347 matches)

Holdout covers WC 2022 + AFCON 2024 + Asian Cup 2024 + Gold Cup 2023 + Copa América 2024 + EURO 2024 + Gold Cup 2025 + AFCON 2025 (group stage). Models marked ★ are on the live production roster.

| Model | Holdout RPS |
|---|---|
| xgboost ★ | **0.18289** ← champion |
| bayesian_poisson ★ | 0.18316 |
| negbin_glm | 0.18373 |
| poisson_glm ★ | 0.18389 |
| random_forest | 0.18392 |
| sarimax | 0.18583 |
| ridge | 0.18813 |
| lstm | 0.19299 |
| cnn | 0.20916 |
| mean_rate_poisson ★ | 0.22872 ← baseline |

`poisson_glm` is preferred over `negbin_glm` despite near-identical RPS (rankings swap depending on random seed); the independent bivariate Poisson formulation also has stronger footing in the football prediction literature (Karlis & Ntzoufras, 2004).

For context: Poisson ranking models on national teams achieve RPS ~0.165; bookmakers on World Cup matches ~0.188–0.194 (Ley et al. 2019; Groll et al. 2019). Note these benchmarks use different training regimes and are not directly comparable — they provide ceiling context only.

Live WC 2026 monitoring results will be recorded in [`docs/notes/wc_live.md`](docs/notes/wc_live.md) throughout the tournament.

---

## Stack

| Layer | Tools |
|---|---|
| Data ingestion | API-Football, Elo ratings (eloratings.net) |
| Feature engineering | pandas, numpy |
| Modeling | scikit-learn, statsmodels, xgboost, PyMC / JAX, Keras, Optuna |
| Experiment tracking | MLflow (remote: DagsHub) |
| Data versioning | DVC (remote: DagsHub) |
| Deployment | Docker, GCP Cloud Run Jobs, GCP Cloud Scheduler |
| Dashboard | Streamlit |
| Testing | pytest |
| Code quality | ruff |

---

## Project structure

```
src/
├── ingestion/       raw data acquisition (API-Football fixtures + stats, Elo TSVs)
├── silver/          cleaning, standardization, Elo join, competition mapping
├── gold/            match-level feature engineering (rolling, temporal, context)
├── models/          training pipeline, 11 candidate models, MLflow utilities
├── inference/       champion/shadow prediction, Monte Carlo tournament simulation
├── monitoring/      live RPS/NLL scoring against settled WC 2026 matches
├── pipeline/        daily/every-2h orchestration trigger
└── dashboard/       Streamlit app — predictions, group standings, knockout paths

data/
├── raw/             immutable Bronze (DVC-tracked)
├── silver/          cleaned Parquet (DVC-tracked)
├── gold/            modeling-ready features, 1 row/match (DVC-tracked)
├── mappings/        team code mappings (baked into Docker image)
└── tournament/      WC 2026 bracket definition (baked into Docker image)

docs/
├── notes/           decisions log, live results, thesis writing notes
├── plans/           thesis roadmap, go-live runbook
├── literature/      paper summaries (Ley 2019, Groll 2019, Maher 1982)
└── figures/         architecture diagrams
```

---

## How to run locally

### Prerequisites

- Python 3.11
- **macOS only:** Xcode (not just Command Line Tools) is required for PyMC/PyTensor C compilation. Install from the App Store, then run:
  ```bash
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  ```
- API-Football key ([api-football.com](https://www.api-football.com/)) — free tier covers the request volume for this project
- DagsHub account for MLflow tracking and DVC artifact remote (or swap in a local MLflow server and local DVC remote)

### Setup

```bash
# 1. Clone and install
git clone https://github.com/tomppa999/MLOps_FootballWCPredictor.git
cd MLOps_FootballWCPredictor
pip install -e ".[dev]"          # or: conda env create -f environment.yml

# 2. Configure credentials
cp .env.example .env
# Edit .env — fill in API_FOOTBALL_KEY and DagsHub credentials

# 3. Pull versioned data artifacts
dvc pull

# 4. Silver and Gold arrive with the pull above — the `silver` and `gold`
#    stages are frozen, so there is nothing to rebuild here. Do NOT run a bare
#    `dvc repro`: the thesis reconstruction stages depend on src/gold/,
#    src/inference/, src/models/, src/monitoring/ and src/silver/, so any local
#    edit there triggers a 4 h 40 rebuild. See "Reproduce the thesis analysis".

# 5. Run the full pipeline (ingest → features → train → inference → simulate)
python -m src.pipeline.trigger

# 6. Launch the dashboard
streamlit run src/dashboard/app.py
```

### Running tests

```bash
pytest
```

### Local-only mode (no cloud)

The pipeline runs fully locally without GCP. Set `SKIP_DVC_PULL=1` if you have data mounted locally and want to skip the DVC pull step. MLflow logs to `./mlruns` by default if `MLFLOW_TRACKING_URI` is not set.

---

## Data and artifact versioning

Data (Bronze/Silver/Gold) and trained model artifacts are versioned with DVC. The remote is hosted on DagsHub alongside the MLflow experiment registry.

- `dvc push` / `dvc pull` — sync artifacts with the DagsHub remote
- `dvc.yaml` holds the Silver/Gold build plus the thesis reconstruction stages. `silver` and `gold` are `frozen: true`, so a `dvc repro` never rebuilds them; pull them instead
- `dvc repro figures` — re-derive the whole thesis analysis (see below). Force individual stages with `dvc repro -f -s <stage>`; a bare `dvc repro -f` propagates upstream and costs 4 h 40
- `data/raw.dvc`, the nested `data/reconstruction/inputs.dvc` / `data/analysis/live.dvc` sidecars and `dvc.lock` are committed to Git; the actual data files are gitignored

---

## Reproduce the thesis analysis

The post-tournament analysis behind thesis section D.1 re-derives offline from DVC-tracked inputs. One command rebuilds all of it: the three research-question datasets (`rq1_matches.csv`, `rq2_entropy.csv`, `rq3_calibration.csv`), the round leaderboards, the faithfulness audit, and the two thesis figures.

### Prerequisites

- The environment, either way round: `conda env create -f environment.yml` (creates `modelops`, Python 3.14) or `pip install -e ".[dev]"` into an existing Python ≥ 3.11.
- DagsHub credentials for the DVC remote — a username and a token. No API-Football key is needed: the analysis never calls the live API.
- ~350 MB of disk for `data/reconstruction`, plus Bronze/Silver/Gold.

### Run it

```bash
# 1. Clone the thesis branch. The `-b` matters: the default branch (main) is
#    stale and its data hashes were never pushed, which surfaces later as a
#    confusing "missing cache files" error.
git clone -b thesis https://dagshub.com/tomppa999/MLOps_FootballWCPredictor.git
cd MLOps_FootballWCPredictor

# 2. Point DVC at the remote with your own credentials (writes .dvc/config.local,
#    which is gitignored).
dvc remote modify --local origin auth basic
dvc remote modify --local origin user <dagshub-username>
dvc remote modify --local origin password <dagshub-token>

# 3. Pull the inputs and re-derive everything.
dvc pull && dvc repro figures
```

`dvc pull` checks out ~19,080 files. `dvc repro figures` then reports *"didn't change, skipping"* for every stage, because the outputs in the pull set already match — that is the expected result, not a failure to do work. To watch the analysis actually run, force the cheap tail, which takes about a minute:

```bash
dvc repro -f -s rq_datasets audit leaderboards figures
```

Always pass `-s` when forcing. A bare `dvc repro -f` propagates upstream into the strand stages, which is a 4 h 40 rebuild (Strand 1 17 min, Strand 2 2 h 01, Strand 5 2 h 37).

### What "same numbers" means

Two built-in checks, both of which passed on a fresh clone on Sep 13:

- `git diff --quiet dvc.lock` and an empty `dvc status` — the re-derived outputs hash to what is committed.
- The audit report at `data/reconstruction/audit_report.json`: **0 fails**. It currently reads 67 pass / 10 warn / 0 fail across 77 checks. Judge it on `n_fail`, not on the warns — all 10 warns are expected, and none is a numerical disagreement: 7 are visibility limits in the offline audit (its monitoring export covers 54 of 104 matches), 2 are the cadence-asymmetric cycle counts left by live's silent shadow skips, and 1 is entropy monotonicity under Monte Carlo noise (worst upward jump 0.0048 nats at 10k simulations).

The load-bearing check is `s5.champion_identity`: the frozen champion (`wc_production` v15) was never touched by any of the bugs the reconstruction had to fix, so the replay must reproduce its logged λ values on all 423 frozen cycles. It does, at max |Δλ| = 4.441e-16. That check diffs `data/reconstruction/inputs/logged_predictions/all_models.parquet` against `data/reconstruction/strand5_frozen_entropy/predictions.parquet` (1.95 M rows) — **trimming either path out of the pull set silently removes the only evidence that the reconstruction is faithful**, and the audit will still report 0 fails.

One known cosmetic exception: `docs/figures/entropy-trajectory.png` does not come out byte-identical across working directories. The whole plot shifts ~5 px vertically from a `tight_layout` difference; every series, glyph and input is otherwise identical. It is `cache: false`, so it never dirties `dvc.lock`.

### Run the tests

```bash
pytest tests/analysis -q   # 63 tests covering the reconstruction
pytest                     # full suite, ~5 min
```

### What is not reproducible

The live run itself. The every-2h Cloud Run cycles ingested from a paid API against a mutable fixture list, and the champion/shadow models were refitted as the tournament progressed. That history cannot be re-executed. What *is* reproducible is the replay: the exact model versions each cycle served are pinned in `data/reconstruction/inputs/models/manifest.csv`, their artifacts are snapshotted alongside, and the Gold table is read back at the commit that cycle actually served. `snapshot_inputs.py` is the one step that needs online DagsHub access, and its outputs are DVC-tracked so nobody has to re-run it.

Note the deliberate version skew: the pinned models were trained on Python 3.11.15 / scikit-learn 1.9.0 in the Cloud Run image and are replayed on Python 3.14.3 / scikit-learn 1.8.0. Verified harmless — see the header of `environment.yml` and the Strand 5 subsection of [`docs/notes/wc_live.md`](docs/notes/wc_live.md).

---

## Current status

- Pipeline deployed and running on GCP Cloud Run (daily pre-tournament, every 2 hours from June 11)

