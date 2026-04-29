"""One-shot seed for the wc_shadow registered model.

Runs once before the first tournament trigger. Loads Gold, calls
``run_shadow_refit``, and registers eight new ``wc_shadow`` versions —
one per non-champion candidate — using each candidate's Optuna-tuned
``best_params`` from the registry (``wc_staging`` cold-start fallback).

Idempotent: re-running creates additional registry versions, which is
harmless. After the first execution, every Path A (full pipeline) and
Path B (champion refit) keeps ``wc_shadow`` in sync with
``wc_production`` automatically.

Usage::

    # from the project root, with the modelops conda env active
    python scripts/backfill_wc_shadow.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running as a plain script without `pip install -e .`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.data_split import load_gold  # noqa: E402
from src.models.mlflow_utils import setup_mlflow  # noqa: E402
from src.models.pipeline import run_shadow_refit  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    log = logging.getLogger("backfill_wc_shadow")

    setup_mlflow()
    df = load_gold()
    log.info("Loaded Gold (%d rows). Refitting non-champion shadows…", len(df))

    run_ids = run_shadow_refit(df)
    log.info("Registered %d wc_shadow versions: %s", len(run_ids), run_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
