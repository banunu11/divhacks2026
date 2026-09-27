"""Train the production model (the one the API serves).

    python -m ml.train               # default production experiment
    python -m ml.train hgb_no_lags   # promote a different experiment

Pick which model to promote with `python -m ml.evaluate compare` first. This
refits it on ALL data (including the test period) so it's as up to date as
possible, and attaches the held-out scores from its experiment run.
"""
import json
import sys

import joblib

from ml import config
from ml.dataset import load_data, training_rows
from ml.experiments import EXPERIMENTS, PRODUCTION


def main(name: str = PRODUCTION):
    if name not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment '{name}'. See: python -m ml.evaluate list")
    exp = EXPERIMENTS[name]

    metrics_file = config.EXPERIMENTS_DIR / name / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        print(f"{name}: held-out roc_auc={metrics['overall']['roc_auc']}")
    else:
        metrics = {}
        print(f"warning: '{name}' has never been evaluated. Run: python -m ml.evaluate run {name}")

    print("loading data...")
    data = load_data()
    rows = training_rows(data)
    print(f"training {name} on {len(rows):,} rows (all data through {data.end})...")
    model = exp.make_model().fit(rows[exp.features], rows["label"])

    joblib.dump({
        "model": model,
        "name": name,
        "features": exp.features,
        "line_profile": data.line_profile,
        "trained_through": str(data.end),
        "metrics": {"experiment": name, **metrics.get("overall", {})},
    }, config.MODEL_PATH)
    print(f"saved -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
