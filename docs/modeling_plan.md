# Modeling Plan

## Candidate models
1. Poisson
2. Negative Binomial
3. XGBoost

## Purpose of each
- Poisson: baseline football count model
- Negative Binomial: alternative count model for overdispersion
- XGBoost: practical ML-style benchmark for tabular prediction

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
### Poisson / Negative Binomial
Initial versions may model home and away goals as separate count targets at the match level.

### Bivariate Poisson
If implemented, bivariate Poisson should be treated as a true joint match-level model that estimates dependence between home and away goals through an additional shared component.
This is model-specific logic and should not be assumed automatically from separate home/away goal models.

### XGBoost
XGBoost may initially be used to predict expected home and away goals through two coordinated regressors or another equivalent match-level formulation.

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