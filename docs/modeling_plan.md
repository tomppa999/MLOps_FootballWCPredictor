# Modeling Plan

## Candidate models

9 models across 5 families are trained in the experimental environment.
All target `home_goals` and `away_goals` as count/distribution outputs.

| # | Model                   | Family        | Library                     |
|---|-------------------------|---------------|-----------------------------|
| 1 | Poisson GLM             | Statistical   | scipy.optimize (custom MLE) |
| 2 | Negative Binomial GLM   | Statistical   | statsmodels                 |
| 3 | Bayesian Poisson (PyMC) | Bayesian      | pymc          |
| 4 | SARIMAX (ARIMAX)        | Time-Series   | statsmodels   |
| 5 | Ridge Regression        | ML baseline   | scikit-learn  |
| 6 | Random Forest           | ML ensemble   | scikit-learn  |
| 7 | XGBoost                 | ML boosting   | xgboost       |
| 8 | LSTM                    | Deep Learning | keras         |
| 9 | 1D CNN                  | Deep Learning | keras         |

## Gold data design
The main Gold table is match-level with one row per match.

## Target philosophy
The main prediction target is goals / score distributions.
Direct match outcomes are derived later from predicted score distributions and Monte Carlo simulation.

## Initial target setup
For the initial implementation, use:
- home_goals
- away_goals

## Model-specific handling

### Poisson GLM (Bivariate Poisson)
Bivariate Poisson model (Karlis & Ntzoufras, 2003) with three latent Poisson
variates: X1 (home-unique), X2 (away-unique), X3 (shared dependence), where
home_goals = X1 + X3 and away_goals = X2 + X3. Each rate is parameterised as
exp(β^T X). Custom MLE via scipy.optimize (L-BFGS-B) with optional L2 penalty.

### Negative Binomial GLM
Two independent statsmodels NegativeBinomial GLMs (home, away) with log link.
Adds a dispersion parameter for overdispersion.

### Bayesian Poisson (PyMC)
Joint generative model with explicit priors over team strength. Expected to
produce better-calibrated uncertainty estimates than frequentist alternatives.

### SARIMAX (ARIMAX)
Uses `home_team_match_index` / `away_team_match_index` as the time axis.
Captures temporal momentum effects not visible to cross-sectional models.

### Ridge Regression
Linear baseline with L2 regularization. Provides a simple, interpretable
benchmark. Requires scaling (see Feature normalization).

### Random Forest
Tree-based ensemble; scale-invariant. Expected to capture non-linear
interactions with no feature engineering overhead.

### XGBoost
Gradient-boosted trees; scale-invariant. Expected to outperform linear models
by capturing non-linear interactions (Elo differential, competition tier, form).
Predict expected home and away goals via two coordinated regressors or an
equivalent match-level formulation.

### LSTM / 1D CNN
Sequence models that consume the rolling match history as a temporal input.
Expected to underperform relative to their complexity given the limited volume
of national team data (see H4 in docs/requirements.md).

## Feature normalization

Normalization is **not** applied in the Gold layer. Gold stores all features at
their natural scale and remains model-agnostic.

Normalization is applied at **training time**, inside each model's pipeline,
fit only on training data to avoid leakage from validation or test rows.

| # | Model                   | Normalization                                                      |
|---|-------------------------|--------------------------------------------------------------------|
| 1 | Poisson GLM             | `StandardScaler` in sklearn `Pipeline` (optimizer convergence)     |
| 2 | Negative Binomial GLM   | `StandardScaler` in sklearn `Pipeline` (optimizer convergence)     |
| 3 | Bayesian Poisson (PyMC) | Manual `StandardScaler` fit on train split before passing to PyMC  |
| 4 | SARIMAX                 | `StandardScaler` on exogenous features before passing to statsmodels |
| 5 | Ridge Regression        | `StandardScaler` in sklearn `Pipeline` (required: L2 penalty treats all feature magnitudes equally without scaling) |
| 6 | Random Forest           | None — tree-based, scale-invariant                                 |
| 7 | XGBoost                 | None — tree-based, scale-invariant                                 |
| 8 | LSTM                    | `StandardScaler` (required for gradient-based learning stability)  |
| 9 | 1D CNN                  | `StandardScaler` (required for gradient-based learning stability)  |

For sklearn-compatible models (1, 2, 5, 6, 7), wrap in an `sklearn.pipeline.Pipeline`
so the fitted scaler + estimator are a single serializable artifact. At inference
time the same pipeline is loaded from MLflow, ensuring the correct transform is
always applied automatically.

For non-sklearn models (3, 4, 8, 9), the fitted scaler must be serialized and
logged to MLflow alongside the model artifact.

## Hyperparameter tuning

### Environment
Tuning is performed in the **Experimental** environment only, before promotion
to QA. Models arrive at QA already tuned. QA is for backtesting only.

The 2022 WC holdout rows must never be seen during tuning. All cross-validation
splits must be strictly time-based (walk-forward) on pre-2022 training data.

### Method
Use **Optuna** (TPE sampler) with walk-forward expanding-window time-series
cross-validation for all 9 models. GLMs and Bayesian Poisson have small search
spaces and use short Optuna studies; ML and deep learning models use longer
studies.

Log every trial's parameters and CV score to MLflow so the full tuning history
is reproducible.

### Key hyperparameters per model

| # | Model                   | Key hyperparameters                                                                 | Search approach         |
|---|-------------------------|-------------------------------------------------------------------------------------|-------------------------|
| 1 | Poisson GLM             | `alpha` (L2 penalty on coefficients)                                                | Short Optuna study      |
| 2 | Negative Binomial GLM   | `alpha` (initial dispersion parameter)                                              | Short Optuna study      |
| 3 | Bayesian Poisson (PyMC) | Prior sigma on team strength coefficients; MCMC draws / tuning steps                | Short Optuna study      |
| 4 | SARIMAX                 | AR/MA order `(p, d, q)` as integers; exogenous feature subsets                      | Optuna over integer orders |
| 5 | Ridge Regression        | `alpha` ∈ [0.001, 1000] log-scale — most impactful parameter                        | Optuna                  |
| 6 | Random Forest           | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`                     | Optuna                  |
| 7 | XGBoost                 | `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_lambda` | Optuna (largest space) |
| 8 | LSTM                    | `units`, `num_layers`, `dropout`, `learning_rate`, `batch_size`                     | Optuna (expensive)      |
| 9 | 1D CNN                  | `filters`, `kernel_size`, `dropout`, `learning_rate`                                | Optuna (expensive)      |

### Notes
- LSTM and CNN trials are expensive; cap the number of Optuna trials (e.g. 20–30)
  and use early stopping within each trial.
- The best hyperparameter config found by Optuna must be logged as MLflow params
  on the final model run so it is fully reproducible.
- Do not tune on the full Gold dataset — always fit/tune on training folds only.

## Common requirements
- train all models through one common pipeline flow where practical
- use the same train/validation/backtest logic
- log parameters, metrics, and artifacts to MLflow
- compare candidates systematically

## Important note on simulation
Do not round expected goals to produce final match predictions.
Use the predicted goal distribution to derive scoreline probabilities and run Monte Carlo simulations.

## Promotion rule
A candidate may be promoted only if:
- data validation passes
- model training succeeds
- evaluation metrics beat the current production baseline on the primary metric
- no critical regression appears on secondary checks

The artifact that gets registered and promoted to Production in the MLflow
model registry is the **refitted** model (trained on the full Gold dataset
available at run time), not the evaluation model. The evaluation run that
justified the selection is linked via an `evaluation_run_id` parameter on the
production-refit run.

## Continuous training strategy

Each retrain cycle follows a two-phase training flow:

1. **Evaluation phase** — train all 9 candidates on pre-WC 2022 data,
   evaluate each on the WC 2022 holdout, log all 9 runs to MLflow tagged
   `stage=evaluation`. Select the best candidate based on holdout metrics.
2. **Production refit phase** — refit the winning model type and
   hyperparameters on the full Gold dataset available at run time (pre-WC
   2022 + WC 2022 + all matches accumulated since, through today). Log as a
   new MLflow run tagged `stage=production-refit`, with an
   `evaluation_run_id` parameter linking back to the evaluation run that
   justified the selection. Register and promote this refitted model to
   Production in the MLflow model registry.
3. **Inference + simulation** — load the Production champion from the
   registry, predict upcoming matches, run Monte Carlo simulation, log
   predictions as MLflow artifacts.

### Pre-tournament (now through June 10)
Daily trigger at 4:30 UTC. When gold grows by 10+ matches since last training
cycle, execute the full two-phase retrain flow above, then inference +
simulation. Expected frequency: 2-3 retrain cycles during international
windows.

### Tournament (June 11 - July 19)
Model is frozen at tournament start. Trigger runs every 30 minutes in
inference-only mode. Load champion from MLflow registry, predict upcoming
matches, run Monte Carlo simulation, log predictions as MLflow artifacts.

### Retraining threshold
10 new gold rows since last training run. Tracked as MLflow run parameter
`gold_row_count`. If gold grew by fewer than 10 rows, skip retraining and
run inference only with the current champion.

### Re-evaluation on WC 2022
Happens every retrain cycle during the evaluation phase. The WC 2022 holdout
metrics are logged on the evaluation runs and preserved as an audit trail,
even though the production-refit model trains on all data including WC 2022.
The champion model type (e.g. XGBoost) is expected to remain stable; the
purpose is drift detection and auditable model selection history.

## Feature set strategy per model family

Rolling shot columns (~1,837–2,068 NaN), tactical rolling columns, and
tactical clusters have NaN for pre-~2020 matches due to sparse statistics
coverage. Rolling goal columns have minimal NaN (~108–109 out of 6,663
rows). Different model families handle this as follows:

| # | Model                   | Feature set  | NaN handling                                               |
|---|-------------------------|--------------|------------------------------------------------------------|
| 1 | Poisson GLM             | Core only    | Shot/tactical features excluded; Core nearly complete       |
| 2 | Negative Binomial GLM   | Core only    | Shot/tactical features excluded; Core nearly complete       |
| 3 | Bayesian Poisson (PyMC) | Core only    | Bayesian models need complete data                         |
| 4 | SARIMAX                 | Core only    | Exogenous matrix must be complete                          |
| 5 | Ridge Regression        | Core only    | StandardScaler cannot handle NaN                           |
| 6 | Random Forest           | Core only    | sklearn RF cannot handle NaN natively                      |
| 7 | XGBoost                 | Full         | Handles NaN natively, learns optimal splits                |
| 8 | LSTM                    | Core or 2020+| Restrict sequences to 2020+ if using tactical              |
| 9 | 1D CNN                  | Core or 2020+| Restrict sequences to 2020+ if using tactical              |

**Core features (8):** `elo_diff`, `competition_tier`, `is_knockout`,
`is_neutral`, `home_team_rolling_goals_for`, `home_team_rolling_goals_against`,
`away_team_rolling_goals_for`, `away_team_rolling_goals_against`. Nearly
complete across all rows (<2% NaN after dropna).

**Full features (28):** Core + rolling shots (6 columns) + rolling tactical
(10 columns) + tactical clusters (4 columns). Substantial NaN in pre-~2020
rows.

## Evaluation metrics

- **Primary (gating):** RPS (Ranked Probability Score) — ordinal-aware, proper scoring rule.
  Computed via analytical Poisson grid: `P(H=h, A=a) = poisson.pmf(h, λ_h) * poisson.pmf(a, λ_a)`,
  truncated at ~10 goals, normalized. Sum over grid to get P(home win), P(draw), P(away win).
  For non-distributional models (Ridge, RF, XGBoost): assume Poisson(predicted_mean).
  Lower is better; 0 = perfect, 1 = worst.
- **Secondary:** RMSE on predicted expected goals.
- **Dropped:** Poisson NLL (only applicable to distributional models, not useful for cross-model comparison)
  and Brier score (redundant with RPS for ordered outcomes).