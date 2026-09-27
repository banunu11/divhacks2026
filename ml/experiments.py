"""Registry of models we can train, evaluate and compare.

To try a new idea, add an entry to EXPERIMENTS below, then:

    python -m ml.evaluate run <name>
    python -m ml.evaluate compare

Each entry says which features the model sees and how to build the estimator.
Any object with fit(X, y) and predict_proba(X) works.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import config
from ml.features import FEATURES

SEED = 42


@dataclass
class Experiment:
    description: str
    features: list[str]
    make_model: Callable[[], object]


# --- simple baselines --------------------------------------------------------
class HistoricalRate:
    """Predicts the training-set delay rate for the same group of rows.

    With group=["line", "dow", "hour"] this is "how often is this line
    delayed at this hour on this day of the week" — the number to beat.
    """

    def __init__(self, group: list[str]):
        self.group = group

    def fit(self, X: pd.DataFrame, y):
        df = X[self.group].assign(_y=np.asarray(y))
        self.rates_ = df.groupby(self.group, observed=True)["_y"].mean().rename("_p")
        self.overall_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame):
        p = X[self.group].join(self.rates_, on=self.group)["_p"].fillna(self.overall_).to_numpy()
        return np.column_stack([1 - p, p])


# --- model factories -----------------------------------------------------------
def hgb(**overrides) -> Callable[[], HistGradientBoostingClassifier]:
    params = dict(
        max_iter=400,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )
    params.update(overrides)
    return lambda: HistGradientBoostingClassifier(**params)


def logreg(features: list[str]) -> Callable[[], object]:
    onehot = [f for f in ["line", "hour", "dow", "month"] if f in features]
    numeric = [f for f in features if f not in onehot]

    def make():
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), onehot),
            ("num", StandardScaler(), numeric),
        ])
        return make_pipeline(prep, LogisticRegression(max_iter=1000))
    return make


# --- feature sets ----------------------------------------------------------------
CALENDAR = ["line", "hour", "dow", "month", "is_weekend", "is_holiday", "is_rush"]
LAGS = [f for f in FEATURES if "delays" in f or f == "hours_since_line_delay"]
WEATHER = config.WEATHER_VARS + ["precip_3h", "snow_24h"]

# --- the registry ------------------------------------------------------------------
EXPERIMENTS: dict[str, Experiment] = {
    "baseline_overall": Experiment(
        "Always predicts the average delay rate for that line",
        ["line"],
        lambda: HistoricalRate(["line"]),
    ),
    "baseline_line_hour": Experiment(
        "Historical rate for this line at this hour of this weekday",
        ["line", "dow", "hour"],
        lambda: HistoricalRate(["line", "dow", "hour"]),
    ),
    "hgb_calendar": Experiment(
        "Gradient boosting on line + calendar only (no weather, no recent delays)",
        CALENDAR,
        hgb(),
    ),
    "hgb_no_lags": Experiment(
        "Gradient boosting without recent-delay features",
        [f for f in FEATURES if f not in LAGS],
        hgb(),
    ),
    "hgb_full": Experiment(
        "Gradient boosting on all features (current production model)",
        FEATURES,
        hgb(),
    ),
    "logreg_full": Experiment(
        "Logistic regression on all features",
        FEATURES,
        logreg(FEATURES),
    ),
}

PRODUCTION = "hgb_full"
