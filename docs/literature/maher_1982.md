# Maher (1982) — Modelling Association Football Scores

**Full reference:** Maher, M. J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.

## Core idea

Goals scored by home and away teams are modelled as Poisson random variables whose means depend on team-specific attacking and defensive strength parameters. Previous work (Moroney 1951, Reep & Benjamin 1968/1971) had rejected the Poisson in favour of the Negative Binomial, but Maher shows this was because those authors pooled all teams together. Once team-specific parameters are introduced, the Poisson fits well.

## Parametrisation

For home team *i* vs away team *j*:

- Home goals: X_ij ~ Poisson(α_i · β_j)
- Away goals: Y_ij ~ Poisson(γ_i · δ_j)

where:
- α_i = home attack strength of team *i*
- β_j = away defense weakness of team *j*
- γ_i = home defense weakness of team *i*
- δ_j = away attack strength of team *j*

Parameters are estimated via iterative maximum likelihood on a full season of league results (22 teams, 462 matches per season). Twelve datasets are analysed: English Football League Divisions 1–4, seasons 1971–72 to 1973–74.

## Hierarchy of models

| Model | Free parameters | What varies across teams | Count (n teams) |
|-------|----------------|--------------------------|-----------------|
| **0** | α, β, γ, δ constant | Nothing — all teams identical | 2 |
| **1A** | α_i free; δ_i = α_i; β, γ constant | Attack only | n + 1 |
| **1B** | β_i free; γ_i = β_i; α, δ constant | Defense only | n + 1 |
| **2** | α_i, β_i free; δ_i = k·α_i, γ_i = k·β_i | Attack and defense (home/away proportional) | 2n |
| **3C** | α_i, β_i, γ_i free; δ_i = α_i | Attack + separate home defense | 3n − 1 |
| **3D** | α_i, β_i, δ_i free; γ_i = β_i | Attack + separate away attack | 3n − 1 |
| **4** | α_i, β_i, γ_i, δ_i all free | Full model | 4n − 2 |

Moving up one level adds (n − 1) parameters. Under the null hypothesis that extra parameters are unnecessary, 2·Δlog-likelihood follows a χ²(n−1) distribution.

## Key findings from the likelihood-ratio tests

Results shown for Division 1, 1971–72 (n = 22, χ²₀.₀₅(21) = 32.7, χ²₀.₀₁(21) = 38.9) and confirmed across all twelve datasets:

| Transition | What is freed | Typical 2·Δlog-L | Significance |
|-----------|---------------|-------------------|--------------|
| 0 → 1A (or 1B → 2) | Attack α_i | 12.5 – 40.6 | **Highly significant** (p < 0.01 in most datasets) |
| 0 → 1B (or 1A → 2) | Defense β_i | 7.1 – 39.9 | **Highly significant** (p < 0.01 in most) |
| 2 → 3D (or 3C → 4) | Away attack δ_i free from α_i | 6.1 – 19.7 | Marginally significant (some at 5%) |
| 2 → 3C (or 3D → 4) | Home defense γ_i free from β_i | 3.9 – 15.1 | **Not significant** |

The order in which parameters are freed has virtually no effect on the Δlog-L contributed by each type — effects are approximately additive.

## Adopted model: Model 2

**Conclusion:** Model 2 is selected as the best trade-off between fit and parsimony.

- Each team has one attack parameter (α_i) and one defense parameter (β_i).
- A team's relative attacking/defensive strength is the same whether playing home or away.
- Home advantage is a single multiplicative constant (k²) applying equally to all teams.
- 2n parameters for n teams (42 for a 22-team league).

## Goodness-of-fit of the independent Poisson (Model 2)

- χ² tests on marginal goal distributions: 19 out of 24 tests are non-significant at 5%.
- Small systematic bias: the model slightly over-predicts 0-goal and 4+-goal events, and under-predicts 1- and 2-goal events. The actual distribution is slightly "narrower" than Poisson.
- This bias is consistent across all twelve datasets but individually too small to reject the model in most cases.

## Bivariate Poisson extension

When the *score difference* (Z = X − Y) is analysed, the independent model under-estimates draws (Z = 0). This suggests positive dependence between home and away scores (e.g., a losing team takes more risks, inflating both teams' scoring).

A bivariate Poisson model is introduced:
- X_ij = U_ij + W_ij, Y_ij = V_ij + W_ij
- U, V, W are independent Poisson with means (μ − q), (λ − q), q respectively
- Correlation ρ ≈ 0.2 across all datasets

| Metric | Independent Poisson | Bivariate Poisson (ρ = 0.2) |
|--------|--------------------|-----------------------------|
| χ² on score difference (Div 1 71–72) | 9.67 | 1.86 |
| Significant cases (out of 12) | 4 at 5% | 0 |

The bivariate model provides a much better fit for the score-difference distribution while retaining the same team parameters from Model 2.

**Important property:** The probability distribution of the *goal difference* (and therefore match outcome probabilities) is identical for both models because the common W term cancels in X − Y. The bivariate extension only matters for predicting exact scorelines, not win/draw/loss probabilities.

## Relevance for our thesis

1. **Baseline model:** Our intercept-only Poisson baseline (grand-mean λ, no team parameters) corresponds to Model 0 — the simplest layer in Maher's hierarchy.
2. **Our full Poisson GLM with team Elo features** approximates Model 2's logic: separate attack/defense contributions captured via Elo-derived features rather than per-team fixed effects.
3. **Independence is fine for RPS:** Since RPS depends on P(home win), P(draw), P(away loss) — which depend only on goal difference — the bivariate extension does not change RPS. Our independent Poisson is sufficient.
4. **Correlation matters for exact scores:** If we later simulate exact scorelines (e.g., for tiebreaker scenarios in Monte Carlo tournament simulation), injecting ρ ≈ 0.2 would improve calibration.
5. **Home advantage as a constant:** Maher finds a single home-advantage multiplier suffices — supports using a binary home/neutral feature rather than per-team home effects.
