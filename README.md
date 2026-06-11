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
pip install -e ".[dev]"

# 2. Configure credentials
cp .env.example .env
# Edit .env — fill in API_FOOTBALL_KEY and DagsHub credentials

# 3. Pull versioned data artifacts
dvc pull

# 4. Rebuild Silver and Gold from raw data
dvc repro

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

- `dvc repro` — rebuild Silver + Gold from raw (runs `build_silver` → `build_gold`)
- `dvc push` / `dvc pull` — sync artifacts with the DagsHub remote
- `data/raw.dvc` and `dvc.lock` are committed to Git; the actual data files are gitignored

---

## Current status

- Pipeline deployed and running on GCP Cloud Run (daily pre-tournament, every 2 hours from June 11)

