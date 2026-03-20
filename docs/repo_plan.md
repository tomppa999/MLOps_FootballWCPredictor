# Repo Plan

## Initial implementation order
1. bootstrap repo structure
2. add small sample raw data and reference mapping files
3. initialize local DVC setup
4. implement Bronze ingestion/parsing for sample inputs
5. implement real historical backfill ingestion for the chosen training window
6. implement Silver cleaning and standardization
7. implement Gold match-level feature skeleton
8. add tests and validation checks
9. implement first model training interface
10. add MLflow
11. add promotion logic
12. add inference and simulation
13. containerize
14. optionally connect DVC remote to DagsHub
15. prepare cloud deployment

## Design rules
- local-first
- small modules
- no premature cloud complexity
- no unnecessary tools
- version code with Git
- version data and key artifacts with DVC
- do not assume DagsHub already exists
- do not assume the real historical dataset already exists before the ingestion pipeline is built