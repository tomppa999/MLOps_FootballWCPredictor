# Modeling Plan

## Candidate models

9 models across 5 families are trained in the experimental environment.
All target `home_goals` and `away_goals` as count/distribution outputs.

| # | Model                   | Family        | Library       |
|---|-------------------------|---------------|---------------|
| 1 | Poisson GLM             | Statistical   | statsmodels   |
| 2 | Negative Binomial GLM   | Statistical   | statsmodels   |
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

### Poisson GLM / Negative Binomial GLM
Model home and away goals as separate count targets at the match level using a
log link. Negative Binomial adds a dispersion parameter for overdispersion.

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
Use **Optuna** (TPE sampler) with walk-forward time-series cross-validation
for all models with a non-trivial search space (Ridge, Random Forest, XGBoost,
LSTM, CNN). For the statistical GLMs and Bayesian Poisson the search space is
small enough for a manual sweep or a short Optuna study.

Log every trial's parameters and CV score to MLflow so the full tuning history
is reproducible.

### Key hyperparameters per model

| # | Model                   | Key hyperparameters                                                                 | Search approach         |
|---|-------------------------|-------------------------------------------------------------------------------------|-------------------------|
| 1 | Poisson GLM             | `alpha` (L2 penalty, if using sklearn `PoissonRegressor`)                           | Manual sweep / short study |
| 2 | Negative Binomial GLM   | Dispersion `alpha` estimated by statsmodels — minimal tuning needed                 | Manual                  |
| 3 | Bayesian Poisson (PyMC) | Prior sigma on team strength coefficients; MCMC draws / tuning steps                | Manual                  |
| 4 | SARIMAX                 | AR/MA order `(p, d, q)` and seasonal order `(P, D, Q, s)`; use AIC/BIC for order selection | Grid or AIC sweep  |
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
- evaluation metrics beat the current production baseline
- no critical regression appears on secondary checks

## Initial evaluation focus
To be finalized later, but likely one primary probabilistic metric plus secondary diagnostics.