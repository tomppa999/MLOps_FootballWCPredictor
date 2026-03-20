# Data Sources

## 1. API-Football
Primary source for:
- fixtures
- match results
- match statistics
- competition/team metadata

## Usage notes
- treat the API as the raw external source
- save raw responses unchanged in Bronze
- do not assume xG is available
- design the feature pipeline around goals and shot/statistics-based features

## 2. EloRatings
Primary source for team Elo history through TSV snapshots.

## Usage pattern
- fetch the relevant team TSV files
- store the raw TSVs in Bronze
- parse them into Silver
- derive the latest valid pre-match Elo in Gold

## Failure handling
If a fresh Elo fetch fails:
- use the last validated snapshot
- log the fallback clearly
- do not silently pretend the refresh succeeded

## Canonical team naming
All source team names should be mapped to a canonical team name and country code using a maintained mapping table.