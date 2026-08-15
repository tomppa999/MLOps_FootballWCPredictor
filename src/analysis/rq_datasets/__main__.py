"""CLI entry: ``python -m src.analysis.rq_datasets`` builds all RQ datasets.

Use ``python -m src.analysis.rq_datasets.export_live`` first (or pass
``--export`` to do both).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.analysis.rq_datasets.build_rq_datasets import build_all
from src.analysis.rq_datasets.export_live import export_live
from src.analysis.rq_datasets.paths import ANALYSIS_ROOT, LIVE_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build RQ-ready analysis datasets from live + reconstruction.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Run live MLflow/Bronze export before building",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="With --export, re-download even if live files exist",
    )
    parser.add_argument("--live-root", type=Path, default=LIVE_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.export:
        export_live(refresh=args.refresh, live_root=args.live_root)
    build_all(live_root=args.live_root, analysis_root=args.analysis_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
