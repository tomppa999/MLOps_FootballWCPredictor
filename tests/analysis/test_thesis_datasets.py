import pandas as pd
import pytest

from src.analysis.thesis_datasets import replace_rows


def test_replace_rows_matches_keys_preserving_order_and_untouched_cells():
    base = pd.DataFrame({"k": ["a", "c", "b"], "p": ["1.00", "0.3", "0.5"], "id": ["q", "r", "s"]})
    updates = pd.DataFrame({"k": ["b", "a"], "p": ["0.45", "0.25"]})
    replace_rows(base, base.k != "c", updates, ["k"], ["p"])
    assert base.to_dict("list") == {
        "k": ["a", "c", "b"],
        "p": ["0.25", "0.3", "0.45"],
        "id": ["q", "r", "s"],
    }
    for keys in (["a"], ["a", "x"], ["a", "a"]):
        with pytest.raises(ValueError):
            replace_rows(base, base.k != "c", pd.DataFrame({"k": keys, "p": "1"}), ["k"], ["p"])
