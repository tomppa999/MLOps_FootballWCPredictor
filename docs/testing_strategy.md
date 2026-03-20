# Testing Strategy

## Goal
Use tests to reduce technical debt and protect the critical pipeline logic.

## Must-test areas
- raw JSON parsing
- raw TSV parsing
- schema validation
- team-name reconciliation
- rolling feature generation
- leakage prevention
- deterministic behavior where feasible
- common model training interface
- promotion logic
- inference schema compatibility

## Test levels

### Unit tests
- small pure functions
- parsers
- mappers
- rolling computations

### Integration tests
- ingestion of sample files
- Silver outputs from Bronze samples
- Gold feature generation on sample data

### Smoke tests
- model training entry points
- MLflow logging hooks, mocked if necessary

## Anti-leakage emphasis
Tests should explicitly verify that target match information is never used in its own features.

## Data validation and expected ranges
In addition to logic tests, validation should check whether important fields fall into plausible ranges or satisfy domain constraints.

Examples:
- possession percentages in [0, 100]
- pass accuracy percentages in [0, 100]
- shots on target less than or equal to total shots
- blocked shots, fouls, cards, saves, and passes are non-negative
- Elo values fall into a plausible historical range
- required columns are present and typed correctly

These checks should be treated as data-quality validation, not only as unit tests.