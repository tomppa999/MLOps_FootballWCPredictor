# Presentation Q&A Prep

## Pipeline Design & Architecture

**Why code-based deployment instead of model-based?**
Code-based means the champion is defined by hyperparameters + training code. New data triggers an automatic refit without manual artifact export. Model-based freezes artifacts at QA time, requiring manual promotion on every data refresh. Code-based is the natural fit for a batch system that runs inference every 30 minutes on 1,128 matchups — there's no serving pressure that would justify frozen, optimized artifacts.

**What Google automation level does this implement?**
Level 1: automated training pipeline with reusable components, DVC-orchestrated rebuilds, scheduled triggers. Full CI/CD of pipeline code (Level 2) is deferred.

**What about IAM / access control?**
Not implemented — single-operator pipeline. DagsHub handles authentication via token for the remote MLflow/DVC store, but there's no role-based access. IAM becomes relevant with multiple team members or shared environments; out of scope for a course project.

**Are you monitoring for data drift?**
No input-distribution monitoring. Goals follow a stable Poisson distribution, Elo values a normal distribution, and context features (competition tier, is_neutral, is_knockout) are deterministic. What we do monitor is performance drift: rolling RPS against WC 2022 baselines.

## Trigger & Retraining Logic

**What triggers the pipeline?**
OR gate: Elo SHA-256 digest change OR new settled API-Football fixtures in a 2-day lookback window. If either source is fresh, the pipeline fires.

**What happens if there is only new Elo data but no new API-Football data?**
Pipeline still fires (OR gate). Gold is rebuilt with updated Elo values for existing matches. But no new match rows are added, so delta = 0 and the dispatch goes to inference-only mode. Net effect: fresh predictions using updated Elo, no retraining.

**If one trigger adds 6 rows and the next adds 6 more, does the champion get refitted?**
Yes. The threshold (≥10 rows) is cumulative since the last refit, not per trigger. The code compares current Gold row count against the `gold_row_count` param stored in the last production MLflow run. After 6+6=12 rows since the last refit, delta=12 ≥ 10 → refit fires.

**Are champion refit runs tracked in MLflow?**
Yes. Every refit creates a new MLflow run tagged `stage: champion-refit`, logs all hyperparameters, forwards the original QA holdout metrics unchanged, records `gold_row_count` and wall time, and registers a new model version with the `champion` alias.

**What is the difference between production-refit and champion-refit tags?**
`production-refit` is the initial deployment — the first time a champion is created after winning the full Experimental → QA → Deploy pipeline. `champion-refit` is every subsequent refit when new data crosses the 10-row threshold — same model, same hyperparameters, just retrained on more Gold rows.

**Why is the model frozen before the tournament and not retrained on tournament matches?**
The tournament is the evaluation target. Retraining on tournament matches would contaminate the test set. It's the same principle as not retraining on holdout data mid-evaluation. Pre-tournament friendlies are different: they're part of the training distribution (tier-4), don't overlap the evaluation window, and update Elo and rolling features — the most important signals.

## Monitoring & Alerts

**What is the monitoring alert threshold and why 130%?**
Alert fires when a model's rolling-mean RPS over the last 24 scored matches exceeds 1.3× its WC 2022 holdout baseline. The top-8 models are within ~2% relative spread, so anything under ~10% would fire on normal noise. 30% is large enough to suppress match-to-match variance over 24 matches, small enough to catch genuine distribution shift.

**Why is the alert window 24 matches?**
One full group-stage matchday: 12 groups × 2 matches. Long enough to suppress early-tournament noise and ensure every team contributes at least once. Short enough to react within the 5-week tournament window.

**What is the operational response to a monitoring alert?**
Alerts are informational only — no automated mid-tournament promotion. The response is: log the alert, check whether degradation is champion-specific or affects all nine shadow models equally. If all degrade together, it's the data (e.g. more upsets than base rate). If only the champion degrades while shadows hold steady, that's a signal to investigate post-tournament. The tournament is too short (~5 weeks) for statistically meaningful retraining.

## Model Selection & Metrics

**Why use two different metrics for tuning and selection (NLL vs RPS)?**
NLL tunes goal-rate accuracy in scoreline space. RPS evaluates W/D/L probability calibration in outcome space. Tuning on RPS directly creates a draw-hedging incentive — draws sit in the middle ordinal position and a wrong draw prediction never gets maximum penalty. By tuning on NLL first, goal-rate parameters are locked in on scoreline accuracy, then RPS selects the champion afterwards. The QA holdout catches any mismatch: the CV NLL winner (NegBin) drops to 7th on holdout RPS.

**If Elo dominates everything, why run nine models?**
We didn't know Elo would dominate this completely before running the experiment. That's the point of a systematic comparison — you need to run it to confirm the hypothesis. Claiming "Ridge on Elo is enough" without evidence would be an untested assumption. The experiment validated that claim.

**Your holdout is only 64 matches — how confident are you that Ridge as champion isn't noise?**
Not very confident, and that's a stated limitation. The top 8 models are within a 0.0046 RPS band, which approaches statistical noise on 64 matches. To strengthen the selection: (a) expand holdout to all continental tournaments since WC 2022, (b) adjust training data composition (e.g. more years, exclude friendlies).

**Why did you select exactly these nine models?**
Representatives of five model families for systematic comparison. Poisson GLM and NegBin follow the natural goal-count distribution. Ridge as ML baseline, Random Forest as simpler tree model, XGBoost as competitive boosting method. SARIMAX tests whether temporal autocorrelation carries signal beyond cross-sectional models. LSTM captures sequential dependencies in fixture history; CNN detects local motifs (e.g. scoring bursts) via 1D convolutions. Both DL models test whether architecture complexity is justified on sparse data.

## Model Implementation Details

**How does the bivariate Poisson GLM use the third parameter (λ3)?**
Fully used. The model fits three log-linear components (λ1, λ2, λ3). Prediction returns (λ1 + λ3, λ2 + λ3) — the shared component adds to both sides' expected goals, inducing score dependence exactly as Karlis-Ntzoufras (2003) specifies.

**Is NegBin overdispersion actually used in prediction?**
The `alpha` parameter is a starting value — statsmodels estimates final dispersion via MLE. So the fitted rates do reflect overdispersion. However, downstream the predicted means are fed into a Poisson grid for scoreline probabilities (same as all models), so overdispersion affects the fitted rate but not the score distribution construction.

**What happened with Bayesian Poisson (H4)?**
H4 was not confirmed, likely due to pipeline design rather than a fundamental limitation of the Bayesian framework. The implementation collapses the posterior to point-estimate lambda means before building the Poisson grid, so the theoretical calibration advantage of full posterior predictive distributions cannot manifest. Full posterior predictive integration (sampling scorelines directly from the posterior) is the immediate next step.

## MLflow Tracking Details

**What gets tracked in MLflow?**
- Experiment: `wc_prediction` (single experiment)
- Registered models: `wc_production` (champion), `wc_staging` (QA candidates), `wc_shadow` (8 non-champion refits)
- Aliases: `champion` on wc_production, `challenger` on wc_staging
- Run tags: `stage` (experimental / qa / production-refit / champion-refit / shadow-refit / inference / monitoring) and `model_name`
- Params: all hyperparameters, `gold_row_count`, `evaluation_run_id`
- Metrics: `cv_nll`, `holdout_rps`, `holdout_nll`, `holdout_rmse_home/away`, wall time; monitoring runs log per-step rps/nll/rmse and cumulative RPS
- Artifacts: importance CSVs (experimental), pyfunc model artifacts (QA/deploy/shadow), `predictions_all_models.csv` (inference), `wc2026_monitoring.csv` (monitoring)
