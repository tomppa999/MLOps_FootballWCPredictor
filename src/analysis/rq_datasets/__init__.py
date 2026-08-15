"""RQ-ready analysis datasets: live export, merge, and loaders.

Thesis analysis should load only via::

    from src.analysis.rq_datasets import load_rq1, load_rq2, load_rq3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame

__all__ = ["load_rq1", "load_rq2", "load_rq3"]


def __getattr__(name: str):
    if name in __all__:
        from src.analysis.rq_datasets import loaders

        return getattr(loaders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
