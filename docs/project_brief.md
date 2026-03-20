# Project Brief

## Objective
Build an end-to-end MLOps pipeline for national-team football match prediction and World Cup tournament simulation.

## Core idea
This project is not just about fitting a model once. It should demonstrate:
- reproducible ingestion
- layered data processing
- time-aware feature engineering
- automated retraining
- model comparison and promotion
- code-based deployment
- testing to reduce technical debt
- explicit versioning of code and important data/artifacts

## Prediction scope
The project targets football match score distributions for national teams, with later support for tournament-level simulation.

## Candidate models
- Poisson
- Negative Binomial
- XGBoost

The best validated model should be promoted to production through an explicit selection rule.

## Target philosophy
Models should predict goals or goal-distribution behavior first.
Match outcomes are derived later from score distributions, not predicted as the main target.

## Gold data design
The Gold dataset should use one row per match, with separate home and away feature columns plus selected difference features.

## Constraints
- keep the architecture simple
- no xG dependency
- no Elo reimplementation
- local-first development
- cloud deployment later
- no DagsHub dependency at the start