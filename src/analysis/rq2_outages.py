"""Public result cutoffs for the four counterfactual RQ2 outage forecasts."""

import pandas as pd

from src.analysis import replay_common as rc

SNAPSHOTS = {
    "r32_pre_japan_brazil": pd.Timestamp("2026-06-29T16:00Z"),
    "r32_japan_brazil_locked": pd.Timestamp("2026-06-29T19:05Z"),
    "r32_germany_paraguay_locked": pd.Timestamp("2026-06-29T23:30Z"),
    "r16_brazil_norway_locked": pd.Timestamp("2026-07-05T22:30Z"),
}

# Tickers publish to minute precision. Use the end of the reported minute
# as a conservative availability bound. Source links are in the README.
_REPORTED_BY = {
    1562344: pd.Timestamp("2026-06-29T19:04Z"),  # Brazil–Japan, full time
    1565176: pd.Timestamp("2026-06-29T23:30Z"),  # Germany–Paraguay, final penalty
    1568100: pd.Timestamp("2026-07-05T22:06Z"),  # Brazil–Norway, full time
}


def public_results(timestamp):
    return rc.parse_wc_results_before_kickoff(
        timestamp, settle_delta=rc.SETTLE_DELTA, reported_results=_REPORTED_BY
    )
