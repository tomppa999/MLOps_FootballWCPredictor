# Literature Review — Football Result Modelling

_Summarises relevant prior work and maps each source to the project's
modeling approach. Based on a chronological survey of football result
modelling from the 1960s to the 2020s._

---

## 1. Directly foundational

These studies form the methodological lineage the project builds on.

### Maher (1982)

Independent Poisson model with per-team attack and defence parameters.
The mean-rate Poisson baseline is the intercept-only case of Maher's
framework ("Model 0"). The entire Poisson goal-modeling pipeline —
independent attack/defence rates, Poisson-distributed goals — descends
from this work.

### Reep et al. (1971)

Identified that goals per match may follow a negative binomial rather
than a Poisson distribution, because scoring probability is affected by
match-related factors (e.g. goals scored prior). The project's Negative
Binomial GLM (candidate #2) is a modern version of this idea:
overdispersion relative to Poisson is captured via an explicit
dispersion parameter.

### Dixon & Coles (1997)

Extended Maher's model with a low-score correction (ρ for 0-0, 1-0,
0-1, 1-1) and exponential time-decay on parameters. The low-score
correction is mathematically irrelevant for RPS (it cancels in goal
difference space), so it is not implemented. However, their
**exponential time-decay** is the direct precursor to the A.4
sample-weight scheme, refined by Ley et al. (2019) with a 3-year
half-period. Their treatment of **home advantage** as an explicit model
component maps to the project's `is_neutral` feature and WC 2026
host-nation override.

### Lee (1997)

Poisson regression for the English First Division (1995/96), followed
by 1,000 Monte Carlo season simulations to produce expected points and
standard deviations. Structurally analogous to the project's Monte Carlo
tournament simulation pipeline. Lee's identified limitations —
overestimating strong teams against weak opponents and not accounting
for squad changes — remain relevant and map to the project's own
acknowledged limitations.

### Karlis & Ntzoufras (2003)

Bivariate Poisson distribution with correlated home and away goals.
Explicitly cited in the modeling plan as the basis for the Poisson GLM
implementation (three latent variates: X1 home-unique, X2 away-unique,
X3 shared dependence). Their finding that independent Poisson
underestimates draws is acknowledged in the project's decision to keep
independent Poisson for simulation simplicity while noting the
approximation in limitations.

---

## 2. Highly relevant

These studies validate or contextualise decisions already made in the
project. None require a change in approach.

### Egidi & Torelli (2021)

Formalise the distinction between **goal-based** models (predict the
exact score) and **result-based** models (predict W/D/L directly).
Result-based models are simpler but treat 1-0 and 5-0 identically,
potentially mis-estimating team strength. They also conclude that
predictive performance alone cannot validate model selection.

**Validation:** The project's goal-based approach is already documented
in `prediction_framing.md`. Egidi & Torelli provide the formal taxonomy
to cite when justifying the design. No change needed.

### Hvattum & Arntzen (2010)

Elo-based prediction with ordered logit regression on 30,000+ English
football matches. Two Elo variants (result-only and
goal-difference-adjusted) outperformed naive baselines but were "vastly
inferior" to bookmaker odds.

**Validation:** The project uses Elo as one feature among many (plus
rolling goals, rolling Elo-change, context features), not as the sole
predictor. This feature-based approach is the methodological response to
the limitation Hvattum & Arntzen identify. Their Elo-vs-bookmaker gap
also provides context for the project's literature benchmarks (Groll et
al.'s bookmaker RPS ~0.188–0.194).

### Rue & Salvesen (2000)

Let attack/defence strengths vary over time and introduced a
psychological factor (stronger teams underestimating opponents). Also
noted that goal intensity depends on match state (conceding goals is
demotivating), which violates the independent Poisson assumption.

**Validation:** Time-varying strengths are already handled by rolling
features and the A.4 time-decay weights. The match-state dependence
point is a real limitation of independent Poisson, but it is already
acknowledged in `decisions.md` under "Independent Poisson for
simulation." It belongs in the thesis limitations section, not as a
design change.

### Koopman & Lit (2015)

Bivariate Poisson with a state-space model that continuously adapts
team strengths over 9 Premier League seasons. More elegant than
discrete rolling windows but substantially more complex to implement.

**Validation:** The project's rolling-window + time-decay approximation
is defensible. The thesis question is about retraining cadence, not
about finding the optimal strength-estimation method. Implementing a
full state-space model would add a tenth modeling approach with marginal
benefit for the research question.

### Constantinou & Fenton (2013) — pi-rating

Dynamic rating system with separate home/away values, emphasising that
recent results matter more than older results and that winning matters
more than goal difference. Pi-rating significantly outperformed the Elo
model of Hvattum & Arntzen (2010).

**Validation:** The project captures "recent results matter more" via
rolling features and time-decay (A.4). The `is_neutral` feature
partially captures home/away asymmetry. Implementing a full pi-rating
would require reimplementing a rating formula, which violates the
project's "do not reimplement Elo" constraint. The pi-rating's
superiority over raw Elo provides additional context for why
`rolling_elo_change` (A.2) adds value beyond absolute Elo.

### Constantinou et al. (2012, 2013)

Bayesian network models with subjective factors (expert-assessed team
strength, psychology, fatigue). Found that "subjective information
tremendously improved the forecasts" and matched bookmaker odds.
Suspected that bookmakers use information not capturable from standard
public data.

**Validation:** The project is deliberately objective and reproducible —
no expert judgment. This is a design choice with known costs. Their
finding about the bookmaker information advantage explains part of the
ceiling gap discussed in `prediction_framing.md`. It belongs in the
thesis discussion as an acknowledged constraint, not as a reason to
introduce subjective inputs.

### Boshnakov et al. (2017)

Challenged the Poisson assumption by using Weibull inter-arrival-times
with a copula for the bivariate goal distribution. Found superior fit
for Premier League match outcomes compared to independent Poisson.

**Validation:** The project assumes Poisson throughout by design.
Adding a Weibull count model would introduce a new candidate family
requiring new evaluation infrastructure. The marginal insight for the
retraining-cadence research question is near zero. Worth citing as
evidence that the Poisson assumption is a known simplification.

### Baio & Blangiardo (2010)

Bayesian approach for Serie A that "better adapted to different dynamics
throughout the season" compared to bivariate Poisson. Outperformed
Karlis & Ntzoufras (2003) on Italian data.

**Validation:** Provides context for the project's Bayesian Poisson
candidate (#3). Their finding that adaptability matters is one
motivation for RQ1 (does per-round retraining improve accuracy?).

---

## 3. Useful contrast

Worth citing in the thesis literature review for context or to
acknowledge alternative approaches, but not directly applicable to the
project's methodology.

### Partida et al. (2021)

xG-based Poisson models (plain xG, scaled xG adjusting for defensive
strength, adjusted xG with home advantage). Methodologically close to
this project — Poisson, context-adjusted, outcomes derived from goal
distributions — but uses xG as the primary input. The project explicitly
excludes xG (not offered by the data source). Cite to note that
xG-based Poisson models exist and that Elo-based features serve as the
alternative.

### Constantinou et al. (2012) — betting profitability

The 5% discrepancy threshold for positive expected return provides a
useful reference point for what "useful prediction" means in the
football context, even though the project does not target betting
profitability.

---

## 4. Out of scope

Methodologically distant from the project's goal-based, feature-driven,
Poisson-family approach. Mentioned for completeness.

| Source | Approach | Why out of scope |
|---|---|---|
| Hill (1974) | Expert panel forecasting | No quantitative model |
| Rotshtein et al. (2005) | Fuzzy model + genetic/neural tuning | Different paradigm entirely |
| Joseph et al. (2006) | Bayesian network for one team (Tottenham), W/D/L | Single-team focus, result-based |
| Owramipur et al. (2013) | Bayesian network for one team (Barcelona), W/D/L | Single-team focus; 92% accuracy is misleading (best team in league, one season) |
| Godin et al. (2014) | Twitter volume + sentiment + fan predictions | Different data paradigm (social media) |
| Schumaker et al. (2016) | Twitter sentiment analysis | Different data paradigm (social media) |
| Goes et al. (2019) | Position tracking data (zone, balance, space mobility) | Requires granular spatial data not available at national-team level |
| Rein et al. (2017) | Passing effectiveness via Voronoi diagrams | Requires tracking data; out of scope |

---

## Thesis placement

| Thesis section | Sources to cite |
|---|---|
| Poisson goal-modeling lineage | Maher (1982) → Dixon & Coles (1997) → Karlis & Ntzoufras (2003) → Ley et al. (2019) |
| Goal-based vs result-based framing | Egidi & Torelli (2021) |
| Time-varying strength / decay | Rue & Salvesen (2000), Koopman & Lit (2015), Dixon & Coles (1997) |
| Elo as feature proxy | Hvattum & Arntzen (2010), Constantinou & Fenton (2013) |
| NegBin overdispersion | Reep et al. (1971) |
| Monte Carlo simulation precedent | Lee (1997) |
| Bayesian adaptability | Baio & Blangiardo (2010), Constantinou et al. (2012, 2013) |
| Subjectivity / expert judgment gap | Constantinou et al. (2012, 2013) |
| Limitations: Poisson assumption | Boshnakov et al. (2017) |
| Limitations: no xG | Partida et al. (2021) |
