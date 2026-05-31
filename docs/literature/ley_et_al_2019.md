# Ley, Van de Wiele, Van Eetvelde (2019)

**Title:** Ranking soccer teams on the basis of their current strength:
A comparison of maximum likelihood approaches

**Published:** Statistical Modelling, 19(1), 55–77

**DOI:** 10.1177/1471082X18817650

---

## Summary

Compares 10 strength-based statistical models for ranking soccer teams by
current strength. Models are grouped into four families: Thurstone-Mosteller,
Bradley-Terry, Independent Poisson, and Bivariate Poisson. All parameters are
estimated via weighted maximum likelihood with two weight types: a smooth
exponential time-decay function (controlled by a "Half Period" parameter) and
a match-importance factor (1 for friendlies, 2.5 for qualifiers, 3 for
continental tournaments, 4 for World Cup matches).

Evaluation metric: RPS (Rank Probability Score).

---

## Key Results

### Model comparison (national teams, 2008–2017, non-friendly matches)

| Model | Optimal Half Period | RPS |
|-------|--------------------:|------:|
| Bivariate Poisson (1 strength/team) | 3 years | 0.1651 |
| Independent Poisson (1 strength/team) | 3 years | 0.1653 |
| Independent Poisson (att+def) | 3.5 years | 0.1656 |
| Bivariate Poisson (att+def) | 3 years | 0.1656 |
| Thurstone-Mosteller | 3.5 years | 0.1658 |
| Bradley-Terry | 4 years | 0.1659 |
| BT-Davidson | 4 years | 0.1660 |
| TM + Goal Difference | 3.5 years | 0.1672 |
| BT + Goal Difference | 3 years | 0.1674 |
| BTD + Goal Difference | 3.5 years | 0.1681 |

### Premier League (2008–2017)

Best model: Bivariate Poisson (1 param/team), Half Period = 390 days,
RPS = 0.1953.

---

## Relevant Findings for Thesis

1. **Poisson models dominate.** They outperform all outcome-based models
   (BT, TM) at both domestic and national team level. Supports the choice
   of goal-modeling over direct W-D-L classification.

2. **Parsimony wins.** 1-parameter-per-team models beat attack/defense
   split models (2 params/team). More parameters ≠ better predictions.

3. **Half Period = 3 years for national teams.** Optimal time-weighting
   gives half-weight to matches 3 years old. National football evolves
   slowly.

4. **Bivariate vs Independent Poisson: negligible difference.** RPS
   0.1651 vs 0.1653. The covariance parameter λ_C is close to zero.
   Justifies using independent Poisson in simulation.

5. **Time depreciation is critical.** All models use weighted MLE with
   exponential decay. Without it, old matches (irrelevant to current
   strength) pollute the estimates.

6. **Match importance weighting helps.** Friendlies weighted 1×, World
   Cup 4×. Differential weighting by competition tier is standard.

---

## Baseline Implications

The paper does NOT use a parameter-free constant-rate baseline. Their
simplest model is the Independent Poisson with 1 team-specific strength
parameter — already much richer than a symmetric mean-rate Poisson.

The Poisson model parameterisation is:
- λ_home = exp(c + (r_i + h) - r_j)
- λ_away = exp(c + r_j - (r_i + h))

where r_i, r_j are team strength parameters, h = home effect, c = intercept.
An intercept-only version (all r_i = 0, h = 0) would reduce to a symmetric
mean-rate model.
