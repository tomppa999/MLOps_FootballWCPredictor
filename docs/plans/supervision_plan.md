# Thesis Supervision Plan

**Student:** Tom Farnschläder  
**Degree programme:** Tampere University  
**Writing language:** English  
**Estimated completion date:** January 2027  
**Target grade:** 5 (excellent)

---

## Supervisors

| Role | Name | Affiliation |
|------|------|-------------|
| Principal supervisor | Sergio Moreschini | Tampere University |

**Examiners:** To be agreed (at supervision plan meeting or later).

**External supervisor CV:** To be attached separately (CV or LinkedIn profile with qualifications and career history).

---

## Title

**The effect of model update frequency on calibration and uncertainty resolution in multi-round probabilistic prediction**

The title may be revised later provided the actual content of the thesis remains within the same subject area.

---

## Research Questions

The thesis is **research-type** with a **hybrid constructive component** (building and deploying an MLOps pipeline). Research questions are presented hierarchically.

**Main research question:**  
How does model retraining cadence affect predictive performance and uncertainty dynamics in a multi-round probabilistic tournament prediction system?

**Sub-questions:**

1. **RQ1 (Accuracy):** Does per-round retraining improve predictive accuracy (RPS, NLL) compared to a frozen pre-tournament model, and does the improvement vary across tournament phases (group stage vs knockout)?

2. **RQ2 (Uncertainty):** How does retraining cadence affect the rate and pattern of uncertainty resolution in tournament advancement probabilities, measured via Shannon entropy trajectories?

3. **RQ3 (Concept drift):** To what extent does concept drift manifest within a single 5-week tournament window, and does per-round retraining mitigate its effects on calibration?

**Empirical use case:** 2026 FIFA World Cup (48 teams, expanded format, 7 natural retraining boundaries).

**Experimental factor:** Two modes run in parallel — frozen (pre-tournament snapshot, never refitted) vs per-round (refitted at each matchday boundary).

**WC 2022 role:** Frozen-model retrospective only (descriptive context, not a manipulated condition).

---

## Methodology

### Constructive component

Design and implementation of a reproducible, local-first MLOps pipeline for probabilistic football prediction:

- Bronze / Silver / Gold data architecture with strict time-aware, leakage-safe features
- Goal-distribution models (Poisson GLM, Negative Binomial GLM, XGBoost, Bayesian Poisson, and others)
- MLflow experiment tracking, model registry, and versioning
- Automated inference, tournament simulation (Monte Carlo), and monitoring
- Code-based deployment on GCP (Cloud Run Job, Cloud Scheduler, DVC remote)
- Git for code versioning, DVC for data and artifact versioning

### Research / experimental component

- **Controlled experiment:** Two retraining modes (frozen vs per-round) run side by side on the same live tournament data (WC 2026, June–July 2026)
- **Evaluation metrics:** Rank Probability Score (RPS), Negative Log-Likelihood (NLL), Shannon entropy of advancement probabilities
- **Baseline:** Symmetric mean-rate Poisson (intercept-only case of Maher 1982) as a model-level threshold; naive-random RPS (0.235) as a monitoring alert floor
- **Training weights:** Exponential time-decay and match-importance sample weights following Ley et al. (2019)
- **Retrospective analysis:** WC 2022 replayed under frozen-model conditions for descriptive comparison
- **Statistical comparison:** Mode performance compared across tournament phases and over time

### Data

| Split | Description | Approx. size |
|-------|-------------|--------------|
| Training | International matches 2008–2022, weighted by recency and importance | ~6,000 matches |
| Holdout | WC 2022 + continental tournament finals 2022–2025 | ~310 matches |
| Live test | WC 2026 | ~104 matches, 7 matchdays |

---

## Preliminary Table of Contents

1. **Introduction** (1.5–3 pages, no subsections)
   - Background and motivation
   - Research questions
   - Structure of the thesis

2. **Background** (8–12 pages)
   - 2.1 Probabilistic prediction in football
   - 2.2 Goal-count models (Poisson family, tree-based methods)
   - 2.3 Retraining and concept drift in sequential prediction
   - 2.4 MLOps: principles and relevance to ML experiments

3. **Methodology** (5–8 pages)
   - 3.1 Experimental design (frozen vs per-round)
   - 3.2 Evaluation metrics (RPS, NLL, Shannon entropy)
   - 3.3 Pipeline architecture and implementation method
   - 3.4 Data sources and feature engineering

4. **System Design and Implementation** (8–12 pages)
   - 4.1 Data pipeline (Bronze / Silver / Gold)
   - 4.2 Model training and selection
   - 4.3 Inference and tournament simulation
   - 4.4 Retraining trigger and dual-mode dispatch
   - 4.5 Deployment (GCP Cloud Run, Scheduler, MLflow)
   - 4.6 Monitoring and alerting

5. **Results** (10–15 pages)
   - 5.1 Model selection and baseline comparison
   - 5.2 RQ1: Accuracy comparison (RPS, NLL by phase)
   - 5.3 RQ2: Entropy resolution trajectories
   - 5.4 RQ3: Drift detection and calibration analysis
   - 5.5 WC 2022 retrospective

6. **Discussion** (5–8 pages)
   - 6.1 Interpretation of results
   - 6.2 Limitations and threats to validity
   - 6.3 Generalisability beyond football
   - 6.4 Further work

7. **Conclusions** (2–3 pages, no subsections)

**References**

**Appendices** (feature list, model hyperparameters, deployment configuration)

**Estimated total length:** 45–60 pages (introduction through end of summary).

---

## Supervision Meetings

**Proposed cadence:** Monthly (first week of each month, or by mutual agreement).

| Period | Meetings | Focus |
|--------|----------|-------|
| June 2026 (pre-WC) | 1 | Confirm plan, pipeline readiness before tournament |
| July 2026 (during/post-WC) | 1 | Live monitoring status, data collection completeness |
| Aug–Nov 2026 | 4 (monthly) | Post-WC analysis, iterative chapter writing |
| Dec 2026 | 1 | Full draft review |
| Jan 2027 | 1 | Final revisions, submission |

**Total:** ~8 meetings.

---

## Work Timeline (high level)

| Phase | Period | Deliverables |
|-------|--------|--------------|
| Model fixes and baseline | May–Jun 2026 | Fixed NegBin/Bayesian evaluation, mean-rate Poisson baseline, QA rerun |
| Thesis feature/data changes | Jun 2026 | Rolling Elo-change, expanded holdout, time-decay weights, model selection |
| Cadence infrastructure + GCP | Jun 2026 | Dual-mode MLflow aliases, per-round refit trigger, Cloud Run deployment |
| Live experiment | Jun–Jul 2026 | WC 2026: frozen vs per-round inference and monitoring |
| Analysis and writing | Aug 2026–Jan 2027 | RQ1–RQ3 results, discussion, full thesis draft, submission |

---

## Open Items for Supervision Plan Meeting

- [ ] Confirm or refine title wording
- [ ] Confirm research questions are appropriately scoped
- [ ] Agree on examiners (now or later)
- [ ] Attach principal supervisor CV / LinkedIn for Moodle submission
- [ ] Confirm Moodle submission deadline
- [ ] Confirm monthly meeting schedule (day/time)

---

## Notes for Moodle Submission

Copy the following sections into the Moodle supervision plan form:

- **Title:** The effect of model update frequency on calibration and uncertainty resolution in multi-round probabilistic prediction
- **Language:** English
- **Completion date:** January 2027
- **Target grade:** 5
- **Supervisor:** Sergio Moreschini, Tampere University
- **Research questions:** See section above (main RQ + RQ1–RQ3)
- **Methods:** Hybrid constructive + experimental (see Methodology section)
- **Table of contents:** See Preliminary Table of Contents above
