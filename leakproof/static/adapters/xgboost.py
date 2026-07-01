"""xgboost adapter."""

from __future__ import annotations

from . import FrameworkAdapter


def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="xgboost",
        estimators={"XGBClassifier", "XGBRegressor", "XGBRanker", "Booster"},
        seeded_callables={"XGBClassifier", "XGBRegressor", "XGBRanker"},
    )
