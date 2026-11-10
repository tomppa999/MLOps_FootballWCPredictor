# Thesis Supervision Plan

Student: Tom Farnschläder    
Degree programme: Tampere University    
Writing language: English    
Estimated completion date: December 2026  
Target grade: 5 (excellent)  
Principal supervisor: Sergio Moreschini (Tampere University)  
Examiners: To be agreed

## Working Title

The effect of ML model update frequency on calibration and uncertainty resolution in multi-round probabilistic prediction

## Research Questions

The thesis is research-type with a hybrid constructive component (building and deploying an MLOps pipeline). Research questions are presented hierarchically.

### Main research question 

**How does model retraining cadence affect predictive performance** and uncertainty dynamics in a multi-round probabilistic tournament prediction system?

### Sub-questions

* RQ1 (Accuracy): Does per-round retraining improve predictive accuracy (RPS, NLL) compared to a frozen pre-tournament model, and does the improvement vary across tournament phases (group stage vs knockout)?  
* RQ2 (Uncertainty): How does retraining cadence affect the rate and pattern of uncertainty resolution in tournament advancement probabilities, measured via Shannon entropy trajectories?  
* RQ3 (Concept drift): To what extent does concept drift manifest within a single 5-week tournament window, and does per-round retraining mitigate its effects on calibration?

### Use case

* Empirical use case: 2026 FIFA World Cup (48 teams, expanded format, 7 natural retraining boundaries).  
* Experimental factor: Two modes run in parallel  
  * frozen (pre-tournament snapshot, never refitted)  
  * per-round (refitted at each matchday boundary).  
* WC 2022 role: Frozen-model retrospective (descriptive context, not a manipulated condition).

## Methodology

### Constructive component

Design and implementation of a reproducible, local-first MLOps pipeline for probabilistic football prediction:

* Bronze / Silver / Gold data architecture with strict time-aware, leakage-safe features  
* Goal-distribution models (Bivariate Poisson GLM, Bayesian Poisson, Negative Binomial GLM, XGBoost, Random Forest, Ridge Regression, LSTM, 1D CNN, SARIMAX)  
* MLflow experiment tracking, model registry, and versioning  
* Automated inference, tournament simulation (Monte Carlo), and monitoring  
* Code-based deployment on GCP (Cloud Run Job, Cloud Scheduler, DVC remote)  
* Git for code versioning, DVC for data and artifact versioning

### Research / experimental component

* Controlled experiment: Two retraining modes (frozen vs per-round) run side by side on the same live tournament data (WC 2026, June–July 2026\)  
* Evaluation metrics: Rank Probability Score (RPS), Negative Log-Likelihood (NLL), Shannon entropy of advancement probabilities  
* Baseline: Symmetric mean-rate Poisson (intercept-only case of Maher 1982\) as a model-level threshold; naive-random RPS (0.235) as a monitoring alert floor  
* Training weights: Exponential time-decay and match-importance sample weights following Ley et al. (2019)  
* Statistical comparison: Mode performance compared across tournament phases and over time  
* Reference replay (descriptive): WC 2022 replayed frozen to provide a 32-team baseline for entropy-resolution trajectories

### Data

* Training Split: International matches 2018–2022, weighted by recency and importance (3,903 matches)  
* Holdout: WC 2022 \+ continental tournament finals 2022–2026 (364 matches)  
* Live test: WC 2026 (104 matches, 7 matchdays)

## Preliminary Table of Contents

1\. Introduction (1.5–3 pages, no subsections)

* Background and motivation  
* Research questions  
* Structure of the thesis

2\. Background (8–12 pages)

* Focus on interjection; clustering 1+2 and 3+4  
* Umbrella intro statistics  
* 2.1 Probabilistic prediction in football  
* 2.2 Goal-count models (Poisson family, tree-based methods)  
* Umbrella intro software engineering  
* 2.3 MLOps: principles and relevance to ML experiments  
* 2.4 (continuous training \-\>) Retraining and concept drift in sequential prediction

3\. Methodology (5–8 pages)

* Mention errors, refer to chapter 5 and 6.2  
* 3.1 (DataOps) Data sources and feature engineering  
* 3.2 (DevOps) Pipeline architecture and implementation method  
* 3.3 (ModelOps)  
  * 3.3.1 Experimental design (frozen vs per-round)  
  * 3.3.2 Evaluation metrics (RPS, NLL, Shannon entropy)

4\. System Design and Implementation (8–12 pages)

* 4.1 Data pipeline (Bronze / Silver / Gold)  
* 4.2 Model training and selection  
* 4.3 Inference and tournament simulation  
* 4.4 Retraining trigger and dual-mode dispatch  
* 4.5 Deployment (GCP Cloud Run, Scheduler, MLflow)  
* 4.6 Monitoring and alerting

5\. Results (10–15 pages)

* 5.1 Model selection and baseline comparison  
* 5.2 RQ1: Accuracy comparison (RPS, NLL by phase)  
* 5.3 RQ2: Entropy resolution trajectories  
* 5.4 RQ3: Drift detection and calibration analysis  
* 5.5 Operational reliability  
  * Incident table (date, symptom, root cause, fix, time to recovery, whether predictions were affected)  
  * Few paragraphs on patterns behind them  
  * Fixes

6\. Discussion (5–8 pages)

* 6.1 Interpretation of results  
* 6.2 Limitations and threats to validity  
* 6.3 Generalisability beyond football  
* 6.4 Further work

7\. Conclusion (2–3 pages, no subsections)  
References  
Appendices (feature list, model hyperparameters, deployment configuration)  
Estimated total length: 45–60 pages (introduction through end of summary).

## Supervision Meetings

Proposed cadence: Monthly.

| Period | Meetings | Focus |
| :---- | :---- | :---- |
| June (pre-WC) | 1 | Confirm plan, pipeline readiness before WC |
| July/August (post-WC) | 1 | Data collection completeness & reconstruction work |
| September | 1 | Analysis, writing process |
| October | 1 | Full draft review |
| November | 1 | Final revisions and submission |

