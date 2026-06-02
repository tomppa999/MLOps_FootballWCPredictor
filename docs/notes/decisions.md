# Key Design Decisions

Captures rationale for non-obvious choices. Maps to thesis Chapter 3 (Methodology)
and Chapter 6 (Discussion / Limitations). Add a bullet whenever you make a decision
that won't be obvious from the code alone.

---

## Baseline model: Maher / Dixon & Coles

- Mean-rate Poisson = Maher's Model 0. Defensible, well-cited.
- Dixon & Coles (1997) is the key Maher citation but irrelevant here:
  its low-score correction (ρ for 0-0, 1-0, 0-1, 1-1) only improves
  exact scoreline prediction, not RPS (goal difference distribution is
  identical — bivariate correction cancels in X−Y).
- Ley et al. (2019) IS the national-team successor to Maher/D&C.
  Frame it as a lineage: Maher → Dixon & Coles → Ley et al.
- See also: `docs/literature/prediction_framing.md`

## Half-period: tuned per model via Optuna [1.0, 5.0]

- Ley et al. found 3 years optimal for all Poisson variants on national teams.
- Adding `half_period_years` to Optuna is low overhead (one numpy recompute per trial).
- Report per-model values in thesis methodology. Deviation from 3y is a finding.
- Mean-rate Poisson baseline: fixed at 3 years (no Optuna, known optimum).

## Competition tier: dual role (sample weight + predictor feature)

- As sample weight: tells the optimizer WC matches are more informative signal.
- As predictor feature: lets the model adjust lambda for the match being predicted.
- Complementary, not redundant — they operate at different stages (training vs inference).
- Weights from Ley et al. (1 / 2.5 / 3 / 4) = pre-2018 FIFA methodology.
  Cite as "Ley et al.'s weights", NOT "current FIFA weights" (FIFA changed 2018).

## Tactical features: dropped (A.1)

- Permutation importance negligible across all 9 models.
- ~49% coverage gap biases training toward UEFA/CONMEBOL.
- Kept in GOLD_COLUMNS for transparency; removed from FEATURE_COLUMNS.

## Rolling Elo-change: included despite weak linear signal (A.2)

- Partial correlation r = −0.028, ΔR² = 0.0008 in a linear model. Weak linear signal.
- Included because: full coverage, no cost, potential nonlinear capture by tree models.
- Justified by ablation (test holdout RPS with/without after refit).

## Holdout expansion: continental tournaments 2022–2025 (A.3)

- KS tests: no tournament's goal distribution differs significantly from WC 2022 (all p > 0.6).
- Pooled tier-1 vs tier-2: KS = 0.030, p = 0.999.
- Gold Cup 2023 is a lambda outlier (1.69 vs 1.11–1.34 for others). Kept — acknowledged.
- Expands holdout from 64 → ~347 matches (4.9×). Needed for stable RPS estimation.

## Goal-based prediction vs direct W/D/L

- See `docs/literature/prediction_framing.md` for the full argument.
- Summary: Ley et al. ALSO predict goals (Poisson) and get their best RPS that way.
  Approach is correct. RPS gap explained by missing time-decay, holdout difficulty,
  and feature-based vs parameter-based estimation. Not a modeling failure.

## WC 2022 role: retrospective only, not a manipulated condition

- Using WC 2022 as a second experimental condition (alongside WC 2026) would confound
  format effects with inter-tournament changes. Avoided by design.
- WC 2022 appears in results only as a descriptive frozen-model replay (D.3).

## Independent Poisson for simulation

- Bivariate extension (Maher 1982, Dixon & Coles 1997) matters for exact scorelines
  but not for RPS. For group tiebreakers that depend on goal counts, independent
  Poisson introduces a small approximation. Acknowledged in limitations.
- Decision: keep independent Poisson for simplicity.

## Home advantage for host nations in 2026

- USA, Canada, Mexico play in their home country for group stage.
- `is_neutral` overridden to False for these matches at inference time — not a Gold feature.
- For KO rounds: venue assigned from `data/tournament/wc2026.json` bracket mapping.
