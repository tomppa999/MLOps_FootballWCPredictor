# Groll, Ley, Schauberger, Van Eetvelde (2019)

**Title:** A hybrid random forest to predict soccer matches in
international tournaments

**Published:** Journal of Quantitative Analysis in Sports, 15(4), 271–287

**DOI:** 10.1515/jqas-2018-0060

---

## Summary

Proposes a "hybrid random forest" that combines a random forest (trained on
team-level covariates like GDP, Elo, FIFA rank, squad age, bookmaker odds,
etc.) with Poisson-derived team ability parameters (from Ley et al. 2019's
bivariate Poisson ranking model). The ability estimates are added as an
additional covariate to the random forest.

Evaluated on all matches from FIFA World Cups 2002–2014 (4-fold
leave-one-WC-out CV) and validated on WC 2018 (64 matches as independent
test set).

---

## Methods

### Covariates (17 variables per team)

Economic: GDP per capita, population.
Sportive: ODDSET (bookmaker) win probability, FIFA rank, Elo rating.
Home advantage: host dummy, same-continent dummy, confederation.
Squad structure: max teammates in same club, avg age, CL/EL players,
legionnaires.
Coach: age, tenure, nationality match.

All metric covariates enter as differences between competing teams.

### Ability Parameters (the key innovation)

Estimated from ALL international matches in the 8 years preceding each WC
using the bivariate Poisson model from Ley et al. (2019) with Half Period =
3 years. This gives a single strength parameter r_i per team, estimated via
weighted MLE. This ability estimate is then used as an additional covariate
in the random forest.

### Models Compared

1. **Hybrid Random Forest** — RF + ability parameter as covariate
2. **Random Forest** — RF on covariates only
3. **Ranking** — bivariate Poisson abilities only (no covariates)
4. **Lasso** — L1-penalized Poisson regression on covariates
5. **Hybrid Lasso** — Lasso + ability parameter
6. **Bookmakers** — odds converted to probabilities (benchmark)

---

## Key Results

### WC 2002–2014 (256 matches, leave-one-out CV)

| Method | Likelihood | Class. Rate | RPS |
|--------|-----------|-------------|------|
| Hybrid Random Forest | 0.422 | 0.548 | **0.187** |
| Bookmakers | 0.425 | 0.524 | 0.188 |
| Random Forest | 0.408 | 0.536 | 0.191 |
| Ranking | 0.413 | 0.532 | 0.191 |
| Hybrid Lasso | 0.429 | 0.552 | 0.194 |
| Lasso | 0.422 | 0.524 | 0.199 |

### WC 2018 Validation (64 matches)

| Method | Likelihood | Class. Rate | RPS |
|--------|-----------|-------------|------|
| Hybrid Random Forest | 0.442 | 0.609 | **0.190** |
| Random Forest | 0.430 | 0.609 | 0.193 |
| Ranking | 0.422 | 0.578 | 0.194 |
| Bookmakers | 0.438 | 0.562 | 0.194 |
| Hybrid Lasso | 0.438 | 0.594 | 0.197 |
| Lasso | 0.424 | 0.562 | 0.207 |

---

## Relevant Findings for Thesis

1. **Team abilities are the most important predictor by far.** Variable
   importance analysis shows the Poisson-derived ability parameter far
   outweighs Elo, FIFA rank, ODDSET odds, GDP, age, etc. Elo_pre in our
   models serves a similar role.

2. **Combining ranking signals with ML improves predictions.** Hybrid RF
   beats both pure RF and pure ranking. Supports our approach of using
   Elo as a feature in XGBoost/GLMs rather than either alone.

3. **Bookmaker-level performance is achievable.** Hybrid RF matches
   bookmakers at RPS 0.187–0.190 for World Cup matches. Realistic
   ceiling for what's achievable.

4. **64 WC matches is a viable validation set.** They use WC 2018 as
   independent validation and get meaningful results. Supports using
   WC 2022 as holdout and expanding for stability.

5. **Poisson assumption for simulation.** RF predictions are treated as
   Poisson intensities; match outcomes drawn from independent Poisson
   distributions. Identical to our pipeline architecture.

6. **Lasso (regularized Poisson) underperforms trees.** RPS 0.199 vs
   0.187. Consistent if our Poisson GLM underperforms XGBoost — this is
   expected, not a bug.

7. **Ability parameter >> Elo rating in importance.** Though both capture
   team strength, the ability parameter (fitted via weighted MLE on
   recent matches with time decay) carries more predictive information
   than raw Elo. Elo is a decent proxy but not optimal.

---

## Baseline Implications

No parameter-free baseline is used. The weakest model tested is the Lasso
(L1-penalized Poisson regression), which achieves RPS 0.199 on WC data.
A constant-rate (mean-rate) Poisson would perform worse than all methods
tested here.

## How Ability Differs from Elo

The ability parameter r_i is estimated by fitting a bivariate Poisson model
via weighted MLE on 8 years of all international matches, with:
- Exponential time decay (Half Period = 3 years)
- Match importance weights (friendly=1, qualifier=2.5, continental=3, WC=4)

Elo ratings use a simpler update rule (K-factor adjustment after each match)
and don't explicitly model goal counts or optimize a likelihood. Both capture
team strength, but the Poisson ability is optimized specifically for goal
prediction.
