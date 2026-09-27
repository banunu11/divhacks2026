"""Train the delay model.

    python -m ml.train

Reads data/raw/*, writes models/delay_model.joblib and models/metrics.json.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml import config
from ml.features import CATEGORICAL, FEATURES, build_features, hourly_delay_counts, line_ridership_profile

TEST_START = "2025-08-01"  # hold out the most recent year
SEED = 42


def load_raw():
    raw = config.RAW_DIR
    alerts = pd.read_parquet(raw / "alerts.parquet")
    weather = pd.read_parquet(raw / "weather.parquet")
    stations = pd.read_parquet(raw / "stations.parquet")
    profile = pd.read_parquet(raw / "ridership_profile.parquet")
    return alerts, weather, stations, profile


def make_training_rows(counts: pd.DataFrame, start: str) -> pd.DataFrame:
    """One row per (hour, line) with a randomly sampled forecast horizon."""
    rng = np.random.default_rng(SEED)
    hours = counts.index[counts.index >= pd.Timestamp(start) + pd.Timedelta(days=2)]
    rows = pd.DataFrame({
        "target_ts": np.repeat(hours, len(config.LINES)),
        "line": np.tile(config.LINES, len(hours)),
    })
    rows["horizon"] = rng.integers(1, config.MAX_HORIZON_HOURS + 1, len(rows))
    stacked = counts.loc[hours].stack()
    rows["label"] = (stacked.to_numpy() > 0).astype(int)
    return rows


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
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


def evaluate(y, p, name):
    m = {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "base_rate": round(float(np.mean(y)), 4),
    }
    print(f"  {name:<28} " + "  ".join(f"{k}={v}" for k, v in m.items()))
    return m


def main():
    print("loading raw data...")
    alerts, weather, stations, profile = load_raw()
    counts = hourly_delay_counts(alerts, start=config.TRAIN_START, end=alerts["date"].max())
    line_profile = line_ridership_profile(stations, profile)
    print(f"  {len(counts):,} hours x {counts.shape[1]} lines")

    rows = make_training_rows(counts, config.TRAIN_START)
    print(f"building features for {len(rows):,} rows...")
    df = build_features(rows, counts, weather, line_profile)

    train = df[df["target_ts"] < TEST_START]
    test = df[df["target_ts"] >= TEST_START]
    print(f"  train={len(train):,}  test={len(test):,}")

    print("training...")
    model = make_model().fit(train[FEATURES], train["label"])

    print("test-set results:")
    metrics = {"model": evaluate(test["label"], model.predict_proba(test[FEATURES])[:, 1], "model")}
    # Baseline: historical delay rate for this line at this hour-of-week.
    base = train.groupby(["line", "dow", "hour"], observed=True)["label"].mean().rename("p")
    p_base = test.join(base, on=["line", "dow", "hour"])["p"].fillna(train["label"].mean())
    metrics["baseline_line_hour_of_week"] = evaluate(test["label"], p_base, "baseline (line x hour/wk)")
    metrics["model_by_horizon"] = {
        int(h): round(roc_auc_score(g["label"], model.predict_proba(g[FEATURES])[:, 1]), 4)
        for h, g in test.groupby("horizon")
    }
    print("  ROC AUC by horizon:", metrics["model_by_horizon"])

    print("refitting on all data...")
    final = make_model().fit(df[FEATURES], df["label"])

    joblib.dump({
        "model": final,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "line_profile": line_profile,
        "trained_through": str(counts.index.max()),
        "metrics": metrics,
    }, config.MODEL_PATH)
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
