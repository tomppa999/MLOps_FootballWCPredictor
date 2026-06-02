# Why Goal-Based Prediction Is the Right Approach

_Captured 2026-06-01 — context from supervision meeting and thesis prep._

---

## The question

Should the pipeline predict W/D/L probabilities directly, or predict goals
and derive outcomes from score distributions? The RPS gap vs. Ley et al.
(0.21 vs 0.165) prompted this question.

## Answer: goal-based prediction is correct

**Ley et al. (2019) also predict goals (Poisson) and derive W/D/L from
that.** Their best result (RPS 0.1651) comes from a bivariate Poisson goal
model. Switching to direct W/D/L classification would discard information
contained in goal margins and would almost certainly perform worse.
The modeling approach is not the source of the RPS gap.

---

## What actually explains the RPS gap

### 1. Different evaluation sets (the dominant factor)

Ley et al. evaluate on **all non-friendly national team matches
(2008–2017)**, which includes hundreds of qualification matches. Qualification
matches are systematically easier to predict: strong teams beat weak teams
by consistent margins, the entropy of the outcome distribution is low.

WC 2022 (our holdout) is the opposite: 64 matches among the 32 best teams
in the world, near-parity throughout, high stakes, tournament pressure.
WC matches are the hardest category of national-team match to predict.

A more honest ceiling for WC-only evaluation: Groll et al. (2019) achieve
RPS **0.190–0.194** on WC 2018 (64 matches). Our 0.21 on WC 2022 is within
~0.02 of that, without time-decay weights and on a different holdout.

### 2. Missing time-decay (planned in A.4, not yet implemented)

Ley et al. explicitly show time depreciation is critical. Without it, a
2010 qualifier receives the same training weight as a 2022 WC match.
This dilutes model estimates toward outdated signal. The exponential decay
with a 3-year half period is the single most important ingredient in their
methodology that our pipeline currently lacks.

### 3. Missing match-importance weights (planned in A.4)

Ley et al. upweight WC matches 4× relative to friendlies. Our current
training treats all matches equally. Both this and item 2 are addressed by
Phase 2 / A.4.

### 4. Feature-based vs. per-team parameter estimation

Ley et al. estimate per-team strength parameters via weighted MLE directly
optimised for goal prediction. We use Elo as a proxy. Elo is a reasonable
approximation but not optimised for this task — Groll et al. confirm this:
Poisson-derived ability parameters outperform raw Elo in importance. This
residual gap is expected and fully explainable in writing.

---

## Implication for the thesis

The RPS comparison to Ley et al. belongs in the **Background section** as
ceiling context, with an explicit note that their evaluation set is
fundamentally different from a WC-only holdout. The correct peer comparison
is Groll et al.'s WC-match RPS of 0.190.

After implementing A.4 (time-decay + importance weights), the expected
improvement will close a meaningful part of the remaining gap. Whatever
residual gap remains is:

- Attributable to holdout difficulty (WC-only vs. all non-friendlies)
- Expected given feature-based vs. parameter-based estimation
- Not a modeling failure — it is an explainable methodological difference

The thesis experiment is about **retraining cadence**, not about beating
Ley et al.'s absolute RPS. The models need to be:

1. Better than the mean-rate Poisson baseline (demonstrated)
2. Calibrated enough that frozen vs. per-round retraining produces a
   measurable difference (the cadence experiment)
3. Methodologically sound — goal-based, properly evaluated, time-aware

All three conditions are met or will be met after A.4. The approach is correct.

---

## References

- Ley, Van de Wiele, Van Eetvelde (2019). *Ranking soccer teams on the
  basis of their current strength.* Statistical Modelling, 19(1), 55–77.
- Groll, Ley, Schauberger, Van Eetvelde (2019). *A hybrid random forest to
  predict soccer matches in international tournaments.* JQAS, 15(4), 271–287.
- Maher (1982). *Modelling association football scores.* Statistica
  Neerlandica, 36(3), 109–118.
- Dixon & Coles (1997). *Modelling association football scores and
  inefficiencies in the football betting market.* Applied Statistics,
  46(2), 265–280.
