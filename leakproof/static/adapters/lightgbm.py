"""lightgbm adapter."""

from __future__ import annotations

from . import FrameworkAdapter


def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="lightgbm",
        estimators={"LGBMClassifier", "LGBMRegressor", "LGBMRanker", "Booster"},
        seeded_callables={"LGBMClassifier", "LGBMRegressor", "LGBMRanker"},
    )
