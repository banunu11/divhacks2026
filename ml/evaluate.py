"""Train, test and compare delay models.

    python -m ml.evaluate list                     # models in the registry
    python -m ml.evaluate run hgb_full             # train on past, test on holdout, save
    python -m ml.evaluate run all                  # every model in the registry
    python -m ml.evaluate compare                  # leaderboard of saved runs
    python -m ml.evaluate show hgb_full            # detailed report for one run
    python -m ml.evaluate replay hgb_full 2026-07-15 [--horizon 1]

Every model is trained on data before config.TEST_START and scored on
everything after it (data it never saw). Results go to
models/experiments/<name>/:

    metrics.json            all scores (committed to git so the team can compare)
    by_line.csv             scores per subway line
    model.joblib            the fitted model (gitignored)
    test_predictions.parquet every holdout prediction, for replay (gitignored)
"""
import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ml import config
from ml.dataset import evaluation_rows, load_data, training_rows
from ml.experiments import EXPERIMENTS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# --- scoring ---------------------------------------------------------------------
def score(y, p) -> dict:
    """All the numbers we care about for one set of predictions."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    brier = brier_score_loss(y, p)
    high = p >= config.RISK_HIGH
    two_classes = 0 < y.sum() < len(y)
    return {
        "n": int(len(y)),
        "delay_rate": round(float(base), 4),
        # ranking: does it put delayed hours above non-delayed ones? 0.5 = random
        "roc_auc": round(roc_auc_score(y, p), 4) if two_classes else None,
        # like roc_auc but focused on the delayed hours; random = delay_rate
        "pr_auc": round(average_precision_score(y, p), 4) if two_classes else None,
        # probability accuracy: mean squared error of the %s. lower = better
        "brier": round(brier, 4),
        # % improvement in brier over always guessing the average. >0 = useful
        "brier_skill": round(1 - brier / (base * (1 - base)), 4) if two_classes else None,
        "log_loss": round(log_loss(y, p, labels=[0, 1]), 4),
        # of hours we flag "high risk", how many actually had a delay?
        "high_precision": round(float(y[high].mean()), 4) if high.any() else None,
        # of hours that had a delay, how many did we flag "high risk"?
        "high_recall": round(float(high[y == 1].mean()), 4) if y.sum() else None,
        "high_flagged_pct": round(float(high.mean()), 4),
    }


def calibration(y, p, bins=10) -> list[dict]:
    """When the model says X%, how often did a delay actually happen?"""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return [
        {"n": int(r.n), "predicted": round(r.predicted, 4), "actual": round(r.actual, 4)}
        for r in out.itertuples()
    ]


def segment_labels(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Ways to slice the test set, to find where the model is weak."""
    hour = df["hour"]
    weekend = (df["is_weekend"] == 1) | (df["is_holiday"] == 1)
    return {
        "time_of_day": pd.Series(
            np.select(
                [df["is_rush"] == 1, (hour >= 1) & (hour < 5)],
                ["rush", "overnight"],
                "off_peak",
            ), index=df.index),
        "day_type": pd.Series(np.where(weekend, "weekend/holiday", "weekday"), index=df.index),
        "weather": pd.Series(
            np.select(
                [df["snow_24h"] > 0.5, df["precip_3h"] > 1.0, df["temperature_2m"] > 30],
                ["snow", "rain", "heat"],
                "normal",
            ), index=df.index),
        "line_recent_delay": pd.Series(
            np.where(df["line_delays_3h"] > 0, "delayed in last 3h", "quiet last 3h"), index=df.index),
    }


# --- run an experiment -------------------------------------------------------------
def run(names: list[str], test_start: str, data=None) -> None:
    data = data or load_data()
    print(f"building train (< {test_start}) and test (>= {test_start}) sets...")
    train = training_rows(data, end=test_start)
    test = evaluation_rows(data, start=test_start)
    print(f"  train={len(train):,} rows   test={len(test):,} rows "
          f"({test['target_ts'].min():%Y-%m-%d} -> {test['target_ts'].max():%Y-%m-%d})")

    for name in names:
        exp = EXPERIMENTS[name]
        print(f"\n== {name}: {exp.description}")
        model = exp.make_model().fit(train[exp.features], train["label"])
        p = model.predict_proba(test[exp.features])[:, 1]
        y = test["label"].to_numpy()

        metrics = {
            "name": name,
            "description": exp.description,
            "features": exp.features,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "train_period": [str(train["target_ts"].min()), str(train["target_ts"].max())],
            "test_period": [str(test["target_ts"].min()), str(test["target_ts"].max())],
            "overall": score(y, p),
            "by_horizon": {int(h): score(y[g.index], p[g.index]) for h, g in test.groupby("horizon")},
            "by_segment": {
                seg: {str(k): score(y[g.index], p[g.index]) for k, g in test.groupby(labels)}
                for seg, labels in segment_labels(test).items()
            },
            "calibration": calibration(y, p),
        }
        by_line = pd.DataFrame(
            {line: score(y[g.index], p[g.index]) for line, g in test.groupby("line", observed=True)}
        ).T

        out = config.EXPERIMENTS_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        by_line.to_csv(out / "by_line.csv", index_label="line")
        joblib.dump({"model": model, "features": exp.features, "name": name}, out / "model.joblib")
        test[["target_ts", "line", "horizon", "label"]].assign(p=p.astype("float32")).to_parquet(
            out / "test_predictions.parquet", index=False)

        o = metrics["overall"]
        print(f"  roc_auc={o['roc_auc']}  pr_auc={o['pr_auc']}  brier={o['brier']}  "
              f"brier_skill={o['brier_skill']}  high: precision={o['high_precision']} recall={o['high_recall']}")
        print(f"  saved -> {out}")


# --- reporting ---------------------------------------------------------------------
def load_metrics() -> dict[str, dict]:
    runs = {}
    for f in sorted(config.EXPERIMENTS_DIR.glob("*/metrics.json")):
        runs[f.parent.name] = json.loads(f.read_text())
    return runs


def compare(sort: str) -> None:
    runs = load_metrics()
    if not runs:
        print("No saved runs yet. Try: python -m ml.evaluate run all")
        return
    rows = []
    for name, m in runs.items():
        o = m["overall"]
        rows.append({
            "model": name,
            "roc_auc": o["roc_auc"],
            "pr_auc": o["pr_auc"],
            "brier": o["brier"],
            "brier_skill": o["brier_skill"],
            "high_prec": o["high_precision"],
            "high_recall": o["high_recall"],
            "auc_+1h": m["by_horizon"]["1"]["roc_auc"],
            f"auc_+{config.MAX_HORIZON_HOURS}h": m["by_horizon"][str(config.MAX_HORIZON_HOURS)]["roc_auc"],
            "test_from": m["test_period"][0][:10],
            "trained_at": m["trained_at"][:16],
        })
    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values(sort, ascending=(sort in ("brier",)))
    print(df.to_string())
    if df["test_from"].nunique() > 1:
        print("\n!! runs used different test periods; scores aren't directly comparable")
    print("\nroc_auc/pr_auc/brier_skill/high_*: higher is better.  brier: lower is better.")


def show(name: str) -> None:
    m = load_metrics().get(name)
    if m is None:
        raise SystemExit(f"No saved run '{name}'. Run: python -m ml.evaluate run {name}")
    cols = ["n", "delay_rate", "roc_auc", "pr_auc", "brier_skill", "high_precision", "high_recall"]

    print(f"{name}: {m['description']}")
    print(f"trained {m['trained_at']}   test {m['test_period'][0][:10]} -> {m['test_period'][1][:10]}\n")
    print("OVERALL")
    print(pd.Series(m["overall"], dtype=object).to_string(), "\n")
    print("BY HOURS AHEAD")
    print(pd.DataFrame(m["by_horizon"]).T[cols].to_string(), "\n")
    for seg, parts in m["by_segment"].items():
        print(f"BY {seg.upper()}")
        print(pd.DataFrame(parts).T[cols].to_string(), "\n")
    print("CALIBRATION (when it says 'predicted', delays happened 'actual' of the time)")
    print(pd.DataFrame(m["calibration"]).to_string(index=False), "\n")
    print("BY LINE (worst roc_auc first)")
    by_line = pd.read_csv(config.EXPERIMENTS_DIR / name / "by_line.csv", index_col="line")
    print(by_line[cols].sort_values("roc_auc").to_string())


def replay(name: str, day: str, horizon: int) -> None:
    f = config.EXPERIMENTS_DIR / name / "test_predictions.parquet"
    if not f.exists():
        raise SystemExit(f"No predictions for '{name}'. Run: python -m ml.evaluate run {name}")
    df = pd.read_parquet(f)
    day_ts = pd.Timestamp(day)
    df = df[(df["target_ts"].dt.normalize() == day_ts) & (df["horizon"] == horizon)]
    if df.empty:
        raise SystemExit(f"{day} isn't in the test period for '{name}'.")

    df = df.assign(hour=df["target_ts"].dt.hour, line=df["line"].astype(str))
    df["cell"] = (df["p"] * 100).round().astype(int).astype(str) + np.where(df["label"] == 1, "*", " ")
    grid = df.pivot(index="line", columns="hour", values="cell")
    grid = grid.reindex([l for l in config.LINES if l in grid.index])
    print(f"{name} on {day} ({day_ts:%A}), predicted {horizon}h ahead")
    print("each cell = predicted delay chance %,  * = a delay actually happened\n")
    print(grid.to_string())
    s = score(df["label"], df["p"])
    print(f"\nthat day: {int(df['label'].sum())} delayed line-hours of {len(df)}, "
          f"roc_auc={s['roc_auc']}, high-risk precision={s['high_precision']} recall={s['high_recall']}")


# --- CLI ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+", help="experiment names, or 'all'")
    r.add_argument("--test-start", default=config.TEST_START)
    c = sub.add_parser("compare")
    c.add_argument("--sort", default="roc_auc")
    s = sub.add_parser("show")
    s.add_argument("name")
    rp = sub.add_parser("replay")
    rp.add_argument("name")
    rp.add_argument("day", help="YYYY-MM-DD, inside the test period")
    rp.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    if args.cmd == "list":
        saved = load_metrics()
        for name, exp in EXPERIMENTS.items():
            mark = "saved" if name in saved else "     "
            print(f"  [{mark}] {name:<20} {exp.description}")
    elif args.cmd == "run":
        names = list(EXPERIMENTS) if args.names == ["all"] else args.names
        unknown = [n for n in names if n not in EXPERIMENTS]
        if unknown:
            raise SystemExit(f"Unknown experiment(s): {unknown}. See: python -m ml.evaluate list")
        run(names, args.test_start)
    elif args.cmd == "compare":
        compare(args.sort)
    elif args.cmd == "show":
        show(args.name)
    elif args.cmd == "replay":
        replay(args.name, args.day, args.horizon)


if __name__ == "__main__":
    main()
