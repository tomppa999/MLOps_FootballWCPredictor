"""Orchestrate the WC 2026 D.1 reconstruction pass (all four strands)."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from src.analysis.strand1_frozen_shadow import run_strand1_frozen_shadow
from src.analysis.strand2_brackets import run_strand2_brackets
from src.analysis.strand3_backfill import run_strand3_backfill
from src.analysis.strand4_entropy import run_strand4_entropy

logger = logging.getLogger(__name__)

STRANDS = {
    "1": ("frozen_shadow", run_strand1_frozen_shadow),
    "2": ("brackets", run_strand2_brackets),
    "3": ("backfill", run_strand3_backfill),
    "4": ("entropy", run_strand4_entropy),
}


def verify_replay_tag() -> None:
    """Ensure we are on the wc2026-end-of-tournament tag state."""
    tag = subprocess.check_output(
        ["git", "describe", "--tags", "--exact-match"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if tag != "wc2026-end-of-tournament":
        logger.warning(
            "HEAD is not exactly wc2026-end-of-tournament (got %r). "
            "Proceeding anyway — ensure dvc.lock matches the tag.",
            tag,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run WC 2026 D.1 reconstruction strands.")
    parser.add_argument(
        "--strands",
        default="1,2,3,4",
        help="Comma-separated strand numbers to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--per-round-monitoring",
        type=str,
        default=None,
        help="Path to per_round wc2026_monitoring.csv for strand 3 copies",
    )
    parser.add_argument(
        "--frozen-monitoring",
        type=str,
        default=None,
        help="Path to frozen wc2026_monitoring.csv for strand 3 (optional)",
    )
    parser.add_argument(
        "--skip-tag-check",
        action="store_true",
        help="Skip git tag verification",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.skip_tag_check:
        try:
            verify_replay_tag()
        except subprocess.CalledProcessError:
            logger.warning("Not on an exact tag — use --skip-tag-check to suppress.")

    selected = [s.strip() for s in args.strands.split(",") if s.strip()]
    results: dict[str, str] = {}

    for key in selected:
        if key not in STRANDS:
            logger.error("Unknown strand %r — choose from %s", key, list(STRANDS))
            return 1
        name, fn = STRANDS[key]
        logger.info("=== Running strand %s (%s) ===", key, name)
        if key == "3":
            from pathlib import Path

            results[name] = fn(
                per_round_monitoring_path=(
                    Path(args.per_round_monitoring) if args.per_round_monitoring else None
                ),
                frozen_monitoring_path=(
                    Path(args.frozen_monitoring) if args.frozen_monitoring else None
                ),
            )
        else:
            results[name] = fn()
        logger.info("Strand %s complete: %s", key, results[name])

    logger.info("D.1 reconstruction complete: %s", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
