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
- Use the targeted [thesis analysis workflow](#reproduce-the-thesis-analysis). Pass `-s` whenever forcing a stage to avoid rebuilding the upstream reconstruction
- `data/raw.dvc`, the nested `data/reconstruction/inputs.dvc` / `data/analysis/live.dvc` sidecars and `dvc.lock` are committed to Git; the actual data files are gitignored

---

## Reproduce the thesis analysis

Run `dvc repro thesis_results` for all three research questions. The target
assembles 1,664 match forecasts and 3,412 tournament forecasts from the pinned
archive and the joint Bivariate Poisson replay. RQ1 and RQ3 score 90-minute
outcomes. RQ2 pairs logged cycles and reserves synthetic points for the figures.
The other models reuse archived forecasts for every match and logged cycle.
[Findings and statistical interpretation](docs/notes/findings_memo.md).

### Bivariate Poisson replay

Restore the [pinned thesis inputs](data/thesis_inputs/README.md), activate
`modelops`, and run `dvc repro poisson_joint_validate rq2_public_outages`. This prepares
the historical Poisson components, simulates each distinct tournament input
with its original seed and 10,000 draws, and checks the results. It reads the
archived model parameters offline. These stages are independent of the
historical strand runners.

- `data/analysis/poisson_joint/matches.csv` contains the 208 Poisson replacement
  forecasts, exact 90-minute win/draw/loss probabilities, source information
  and RPS against the archived match labels. The downstream 90-minute analysis
  rescores those probabilities against regulation-time outcomes.
- `data/analysis/poisson_joint/cycles.csv` maps each of the 853 cycle/cadence
  rows to its six entropy values and simulation state. Join on `state_id` to
  `advancement.parquet` in the same directory for team probabilities, or to
  `data/analysis/poisson_joint_inputs/states.json` for the seed and result locks.
- `data/analysis/poisson_joint_checks.json` records the comparison with the
  original-seed sensitivity run, four independent-sampler controls and all
  eight Poisson outage rows.
- `data/analysis/rq2_public_outages.csv` contains the four synthetic snapshots
  for both cadences and all four models (32 rows). Each uses the round seed
  and 10,000 simulations. The three comparison models use their historical
  regime parameters. Each pin is checked against archived forecasts immediately
  before the outage. Every locked knockout winner must advance in all simulations.
  The public cutoffs apply only to the reconstructed RQ2 trajectories. The paired
  RQ2 analysis excludes synthetic observations.

On 29 June the 16:00 UTC forecast precedes Brazil–Japan. At 19:05 Brazil has
advanced, and at 23:30 Paraguay has also advanced. Full time in Brazil–Japan
was reported at 19:03 UTC and the decisive Paraguay penalty at 23:29 UTC.
[Al Jazeera, Brazil–Japan](https://www.aljazeera.com/amp/sports/liveblog/2026/6/29/brazil-vs-japan-live-fifa-world-cup-2026)
and [The Guardian, Germany–Paraguay](https://www.theguardian.com/football/live/2026/jun/29/germany-v-paraguay-world-cup-2026-last-32-live).

On 5 July the 22:30 UTC forecast includes Norway’s win over Brazil. Mexico–England
was still to come. Full time was reported at 22:05 UTC. The replay allows the
reported minute to finish before treating the result as publicly available.
[The Guardian, Brazil–Norway](https://www.theguardian.com/football/live/2026/jul/05/brazil-v-norway-world-cup-2026-last-16-live).

### Restore and reproduce offline

Use the `thesis` revision containing the joint replay, the project's `modelops`
environment and the [local archive](data/thesis_inputs/README.md). The six
DVC input roots below are sufficient in an empty checkout. Replace the example
store path with the directory containing the archived DVC objects. The run
uses the recorded cycle seeds and the project's 10,000-draw tournament setting.

```bash
conda activate modelops
dvc remote add --local thesis-archive /absolute/path/to/data/thesis_inputs/local_store
dvc pull -r thesis-archive data/raw.dvc data/analysis/live.dvc data/reconstruction/inputs.dvc \
  data/thesis_inputs/baseline.dvc data/thesis_inputs/poisson.dvc data/thesis_inputs/sensitivity.tar.gz.dvc
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
dvc repro thesis_results
```

The result verification is in `data/analysis/thesis_results.json`. It records
input identities, exact source-row checks, RQ2 sample membership, Holm family
sizes and output hashes. Repeat the last command to check that the stages skip.
The full reproduction uses numeric historical model states and the archived
data offline. A targeted deterministic replay uses
`dvc repro -f -s poisson_joint_replay` followed by the ordinary target.

- `data/analysis/thesis/` contains the assembled match and trajectory inputs.
  `data/analysis/regulation_time/` contains the 90-minute scores and calibration bins.
- `docs/tables/regulation_time/` contains the RQ1 and RQ3 tables. The RQ2 tables
  and first-refit comparisons are in `docs/tables/rq2_*.csv`.
- `docs/figures/entropy-targets/` contains six stage-specific curves. The
  combined entropy plots are under `docs/figures/`, and the two official
  calibration plots are under `docs/figures/regulation-time/`.
- The local tag `thesis-before-joint-poisson` and the baseline bundle preserve
  the independent-sampling results. The settlement-only comparison and the
  reconstruction audit document that historical analysis.

### Run the tests

```bash
pytest tests/analysis -q
pytest                     # full suite, ~5 min
```

### What is not reproducible

The live run itself. The every-2h Cloud Run cycles ingested from a paid API against a mutable fixture list, and the champion/shadow models were refitted as the tournament progressed. That history cannot be re-executed. What *is* reproducible is the replay: the exact model versions each cycle served are pinned in `data/reconstruction/inputs/models/manifest.csv`, their artifacts are snapshotted alongside, and the Gold table is read back at the commit that cycle actually served. `snapshot_inputs.py` is the one step that needs online DagsHub access, and its outputs are DVC-tracked so nobody has to re-run it.

Note the deliberate version skew: the pinned models were trained on Python 3.11.15 / scikit-learn 1.9.0 in the Cloud Run image and are replayed on Python 3.14.3 / scikit-learn 1.8.0. Verified harmless — see the header of `environment.yml` and the Strand 5 subsection of [`docs/notes/wc_live.md`](docs/notes/wc_live.md).

---

## Current status

- Pipeline deployed and running on GCP Cloud Run (daily pre-tournament, every 2 hours from June 11)

