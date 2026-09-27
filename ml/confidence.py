"""Confidence labels and "clears in X-Y min" ranges for the in-effect forecast.

    python -m ml.confidence

Two additions on top of the in_effect probability from ml.horizons, both
tested on the held-out year:

1. Clearing time, for lines with a delay in effect right now. A model
   predicts P(still delayed h minutes from now) for h = 5..60. Read as a
   curve, it gives the most likely clearing time (where the curve drops to
   50%) and two ranges: the middle half of outcomes (75% -> 25%) and the
   middle 80% (90% -> 10%). A good range contains the real clearing time
   that often, and no more. Compared against a simple baseline: the typical
   remaining duration for that cause of incident, from the training years.

2. A confidence label (high / medium / low) for each in_effect probability,
   based on how accurate the model is in that situation: is a delay already
   ongoing, and how far ahead are we looking.

"Clearing" uses the same assumed end time as the in_effect label: 20 minutes
after the incident's last update (horizons.RECOVERY_MIN).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml import config
from ml import horizons as H
from ml.evaluate import score

CURVE = np.arange(5, 65, 5)      # minutes ahead at which the clearing curve is predicted
OVER = 65                        # stands for "more than an hour"
RANGES = {"middle 50%": (0.25, 0.75), "middle 80%": (0.10, 0.90)}
FEATURES = H.MODELS[H.BEST][0]
OUT_DIR = H.OUT_DIR


def confidence_label(ongoing, horizon) -> np.ndarray:
    """high: delay ongoing, <= 15 min ahead. medium: ongoing <= 30 min, or quiet line <= 15 min. low: the rest."""
    ongoing, horizon = np.asarray(ongoing, bool), np.asarray(horizon)
    return np.where(ongoing & (horizon <= 15), "high",
                    np.where((ongoing & (horizon <= 30)) | (~ongoing & (horizon <= 15)), "medium", "low"))


def is_ongoing(g: H.Grid, cells, lines) -> np.ndarray:
    """A delay is in effect right now: the line's last alert update was <= RECOVERY_MIN ago."""
    return (cells - g.last["any_alert"][cells - 1, lines]) * H.STEP <= H.RECOVERY_MIN


def minutes_to_clear(g: H.Grid, cells, lines) -> np.ndarray:
    """Actual minutes from the start of `cells` until the line's delay ends (OVER if > 60)."""
    in_effect = np.diff(g.cum["in_effect"], axis=0)
    out = np.full(len(cells), OVER)
    for line in range(len(config.LINES)):
        rows = np.flatnonzero(lines == line)
        clear = np.flatnonzero(in_effect[:, line] == 0)
        k = np.minimum(np.searchsorted(clear, cells[rows]), len(clear) - 1)
        minutes = (clear[k] - cells[rows]) * H.STEP
        out[rows] = np.where((minutes > 60) | (minutes < 0), OVER, minutes)
    return out


def sample_ongoing(g: H.Grid, start, end, n_draw, rng):
    cells, lines = H.sample_issues(g, start, end, n_draw, rng, reach_min=max(CURVE) + H.STEP)
    keep = is_ongoing(g, cells, lines)
    return cells[keep], lines[keep]


# --- 1. clearing time ------------------------------------------------------------
def fit_clearing(g: H.Grid, n_draw=4_000_000):
    """Model of P(still delayed h minutes from now), trained on ongoing delays only."""
    rng = np.random.default_rng(H.SEED + 2)
    cells, lines = sample_ongoing(g, config.TRAIN_START, config.TEST_START, n_draw, rng)
    h = rng.choice(CURVE, len(cells))
    rows = H.make_rows(g, cells, lines, h, width=H.STEP)
    y = minutes_to_clear(g, cells, lines) > h
    model = H.lgbm()().fit(rows[FEATURES], y)
    return model, rows.assign(clear_min=minutes_to_clear(g, cells, lines))


def clearing_curves(model, g: H.Grid, cells, lines) -> np.ndarray:
    """P(still delayed at 5, 10, ..., 60 min) per (cell, line); forced to never rise."""
    n, k = len(cells), len(CURVE)
    rows = H.make_rows(g, np.repeat(cells, k), np.repeat(lines, k), np.tile(CURVE, n), width=H.STEP)
    p = model.predict_proba(rows[FEATURES])[:, 1].reshape(n, k)
    return np.minimum.accumulate(p, axis=1)


def quantile(curves: np.ndarray, q: float) -> np.ndarray:
    """Minutes by which the delay has cleared with probability q (OVER if not within the hour)."""
    done = curves <= 1 - q
    return np.where(done.any(axis=1), CURVE[done.argmax(axis=1)], OVER)


def estimates_from_curves(curves) -> pd.DataFrame:
    out = {"mid": quantile(curves, 0.5)}
    for name, (lo, hi) in RANGES.items():
        out[f"{name} lo"], out[f"{name} hi"] = quantile(curves, lo), quantile(curves, hi)
    return pd.DataFrame(out)


def estimates_by_cause(train_rows: pd.DataFrame, causes: pd.Series) -> pd.DataFrame:
    """Baseline: quantiles of remaining time for this cause in the training years."""
    qs = {"mid": 0.5} | {f"{n} {side}": q for n, pair in RANGES.items() for side, q in zip(("lo", "hi"), pair)}
    table = pd.DataFrame({k: train_rows.groupby("inc_cause", observed=True)["clear_min"].quantile(q)
                          for k, q in qs.items()})
    overall = pd.Series({k: train_rows["clear_min"].quantile(q) for k, q in qs.items()})
    return table.reindex(causes.astype(str)).fillna(overall).reset_index(drop=True)


def judge(est: pd.DataFrame, actual: np.ndarray) -> dict:
    out = {"median error (min)": float(np.median(np.abs(est["mid"] - actual))),
           "within 10 min": float((np.abs(est["mid"] - actual) <= 10).mean())}
    for name in RANGES:
        lo, hi = est[f"{name} lo"], est[f"{name} hi"]
        out[f"{name}: contains actual"] = float(((actual >= lo) & (actual <= hi)).mean())
        out[f"{name}: avg width (min)"] = float((hi - lo).mean())
    return out


# --- 2. confidence labels ------------------------------------------------------------
def label_report(g: H.Grid, train, test) -> pd.DataFrame:
    feats, make, _ = H.MODELS[H.BEST]
    p = make().fit(train[feats], train["label_in_effect"]).predict_proba(test[feats])[:, 1]
    test = test.assign(p=p, ongoing=(test["min_since_line_alert"] <= H.RECOVERY_MIN).to_numpy())
    test["confidence"] = confidence_label(test["ongoing"], test["horizon"])

    grid = test.groupby(["ongoing", "horizon"]).apply(
        lambda d: roc_auc_score(d["label_in_effect"], d["p"]), include_groups=False).unstack().round(3)
    grid.index = grid.index.map({True: "delay ongoing now", False: "line quiet now"})
    print("\nroc_auc by situation x minutes ahead (what the labels are based on):")
    print(grid.to_string())

    rows = []
    for label in ["high", "medium", "low"]:
        d = test[test["confidence"] == label]
        s = score(d["label_in_effect"], d["p"])
        called = d["p"] >= 0.5
        rows.append({"confidence": label, "share of forecasts": len(d) / len(test),
                     "roc_auc": s["roc_auc"], "brier_skill": s["brier_skill"],
                     "calls >= 50% that were right": d.loc[called, "label_in_effect"].mean() if called.any() else np.nan,
                     "avg distance from 50/50": float(np.abs(d["p"] - 0.5).mean())})
    return pd.DataFrame(rows)


def run():
    pd.set_option("display.width", 200)
    g = H.load_grid()
    end = g.t0 + pd.Timedelta(minutes=H.STEP * g.n_cells)

    model, train_rows = fit_clearing(g)
    cells, lines = sample_ongoing(g, config.TEST_START, end, 1_500_000, np.random.default_rng(H.SEED + 3))
    actual = minutes_to_clear(g, cells, lines)
    causes = H.make_rows(g, cells, lines, np.full(len(cells), 5), width=H.STEP)["inc_cause"]
    print(f"clearing time: {len(train_rows):,} ongoing delays to train on, {len(cells):,} to test on")
    print("actual minutes to clear in the test year:",
          pd.Series(actual).describe(percentiles=[.25, .5, .75]).round(0)[["25%", "50%", "75%"]].to_dict(),
          f"(over an hour: {(actual == OVER).mean():.0%})")

    model_est = estimates_from_curves(clearing_curves(model, g, cells, lines))
    base_est = estimates_by_cause(train_rows, causes)
    # the in_effect rule itself: no more updates, so it ends RECOVERY_MIN after the last one
    rule = (g.last["any_alert"][cells - 1, lines] + H.RECOVERY_MIN // H.STEP + 1 - cells) * H.STEP
    rule_est = pd.DataFrame({"mid": rule} | {f"{n} {s}": rule for n in RANGES for s in ("lo", "hi")})
    # About half of delays get no further update, so their end is exactly the rule's; the rest are
    # the real prediction problem, so score both.
    longer = actual != rule
    clearing = pd.concat({
        subset: pd.DataFrame({"model": judge(model_est[m], actual[m]),
                              "baseline: by cause": judge(base_est[m], actual[m]),
                              "baseline: ends 20 min after last update": judge(rule_est[m], actual[m])})
        for subset, m in [("all ongoing delays", np.ones(len(actual), bool)),
                          ("delays that got another update and ran longer", longer)]
    })
    print(f"\n{longer.mean():.0%} of ongoing delays got another update and ran longer than the rule says")
    print("how good are the 'clears in X-Y min' estimates? (a middle-50% range should contain the "
          "actual at least 50% of the time, a middle-80% range at least 80%)")
    print(clearing.round(2).to_string())

    train, test = H._samples(g, 1_500_000, 100_000)
    labels = label_report(g, train, test)
    print("\nconfidence labels on the in_effect forecast:")
    print(labels.round(3).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clearing.to_csv(OUT_DIR / "clearing_eval.csv")
    labels.to_csv(OUT_DIR / "confidence_labels.csv", index=False)
    return clearing, labels


if __name__ == "__main__":
    run()
