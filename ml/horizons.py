"""How far ahead can we see? Compare models from 5 minutes to 1 hour ahead.

    python -m ml.horizons                   # train + test every model in MODELS (~5 min)
    python -m ml.horizons lgbm lgbm_text    # just these; other rows in results.csv are kept
    python -m ml.horizons --quick ...       # 5x fewer rows, for trying things out
    python -m ml.horizons --plot            # redraw the chart from results.csv

The hourly pipeline (ml.features / ml.dataset) can only look whole hours
ahead. This one works on a 5-minute grid, and every row asks the same
question at every horizon, so accuracy is comparable across horizons:

    "Standing at time t, will there be a delay on line L in the 15-minute
     window [t + h, t + h + 15 min)?"

for three meanings of "delay":

    any_alert   any delay-alert update about line L in the window (what the
                hourly model and the app call a delay)
    new_delay   a new incident starts affecting line L in the window
    in_effect   an incident is affecting line L at some point in the window.
                The MTA never posts an all-clear, so an incident counts as
                in effect from its first update until RECOVERY_MIN after its
                last one.

The train and test samples are fixed by seed, so every model is scored on
exactly the same rows. Earlier runs out to 6 hours, including recency
weighting and heavier tuning (neither helped), are archived in
results_v2_up_to_6h.csv. Results go to models/experiments/horizons/, figures
to reports/figures/.
"""
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml import config
from ml.evaluate import score
from ml.experiments import HistoricalRate, hgb, logreg
from ml.features import (LINE_TO_IDX, calendar_features, line_ridership_profile,
                         parse_lines, weather_features)

SEED = 42
STEP = 5                                          # minutes per grid cell
WINDOW = 15                                       # width of the window we predict
HORIZONS = [5, 10, 15, 20, 30, 45, 60]            # minutes ahead
LAGS = {"15m": 15, "1h": 60, "3h": 180, "6h": 360, "24h": 1440}
TARGETS = ["any_alert", "new_delay", "in_effect"]
RECOVERY_MIN = 20       # assumed: service is back to normal this long after an incident's last update
INCIDENT_MEMORY = 180   # an update older than this (minutes) no longer describes a "current" incident
OUT_DIR = config.EXPERIMENTS_DIR / "horizons"

BASE = (
    ["line", "horizon", "hour", "time_of_day", "dow", "month", "is_weekend", "is_holiday", "is_rush"]
    + config.WEATHER_VARS + ["precip_3h", "snow_24h", "line_riders"]
    + [f"{scope}_{kind}_{w}" for kind in ("alerts", "new") for scope in ("line", "sys") for w in LAGS]
    + ["min_since_line_alert", "min_since_line_new"]
)
# what the latest alert text says about the line's current incident (see incident_features)
TEXT = ["inc_cause", "inc_ongoing", "inc_major", "inc_age_min", "inc_updates", "inc_lines"]
# recent delays on lines that share stations with this one, weighted by how much they share
NEIGHBORS = ["nbr_alerts_15m", "nbr_alerts_1h", "nbr_alerts_3h", "nbr_new_1h", "nbr_top_active"]
# non-delay alerts (suspensions, skipped stops, reroutes, schedule notices) and daily temperature extremes
CONTEXT = ["line_disrupt_1h", "line_disrupt_6h", "sys_disrupt_1h", "min_since_line_disrupt",
           "line_suspend_1h", "line_schedule_24h", "temp_max_day", "temp_min_day"]
# status labels in the non-delay alerts that describe the timetable rather than a live problem
SCHEDULE_STATUSES = {"weekday-service", "weekend-service", "essential-service", "no-scheduled-service",
                     "sunday-schedule", "saturday-schedule", "planned-work", "special-notice", "station-notice",
                     "information-outage", "arrival-information-outage", "special-event", "on-or-close"}

# --- reading the alert text ------------------------------------------------------
# First match wins, so more specific causes come first.
CAUSES = {
    "person_struck": r"struck|on the track|on the roadbed|in the tunnel|trespass",
    "medical": r"medical|\bems\b|sick|ill |injur|unconscious|someone in need",
    "police": r"nypd|police|disruptive|unruly|fight|dispute|investigation",
    "signal": r"signal",
    "infrastructure": r"switch|track|rail|debris|fire|smoke|power|electric|flood|water|tree|cable|bridge",
    "train_problem": r"brake|mechanical|door|disabled|train that|train with|equipment|removed? a train"
                     r"|train car|vandal|cleaning|from service",
    "crew": r"crew|operator|conductor|staff",
}
CAUSE_NAMES = ["none", "other"] + list(CAUSES)
_ONGOING = re.compile(r"delayed\b.*\bwhile|while (we|crews|emergency|nypd|ems|the)|investigat|we're|we are working")
_RECOVERING = re.compile(r"running with delays\b.*\bafter|after (we|nypd|ems|crews|emergency|an? )|resum"
                         r"|back on schedule|getting trains back|wait longer")
_MAJOR = re.compile(r"severe|suspend|reroute|skipped|express-to-local|local-to-express|cancel")


def cause_of(header: str) -> str:
    h = str(header).lower()
    for name, pattern in CAUSES.items():
        if re.search(pattern, h):
            return name
    return "other"


def phase_of(header: str) -> int:
    """1 = crews still dealing with it, 0 = recovering, -1 = can't tell."""
    h = str(header).lower()
    if _ONGOING.search(h):
        return 1
    if _RECOVERING.search(h):
        return 0
    return -1


# --- 5-minute event grid --------------------------------------------------------
@dataclass
class Grid:
    t0: pd.Timestamp               # start time of cell 0
    n_cells: int
    cum: dict                      # target -> cumulative counts; cum[k] = events in cells < k
    cum_sys: dict                  # target -> same, summed over all lines
    last: dict                     # target -> index of the latest cell <= k with an event
    updates: list                  # per line: that line's alert updates, sorted by cell
    overlap: np.ndarray            # overlap[l, m] = share of line l's stations that line m also serves
    weather: pd.DataFrame
    line_profile: pd.DataFrame

    def cell(self, ts) -> int:
        return int((pd.Timestamp(ts) - self.t0) // pd.Timedelta(minutes=STEP))


def _cells(ts: pd.Series, t0) -> np.ndarray:
    return ((ts - t0) // pd.Timedelta(minutes=STEP)).to_numpy()


def _accumulate(C: np.ndarray):
    idx = np.arange(len(C), dtype=np.int32)[:, None]
    cum = np.vstack([np.zeros((1, C.shape[1]), np.int32), np.cumsum(C, axis=0, dtype=np.int32)])
    last = np.maximum.accumulate(np.where(C > 0, idx, -10**6).astype(np.int32), axis=0)
    return cum, cum.sum(axis=1), last


def load_grid() -> Grid:
    raw = config.RAW_DIR
    alerts = pd.read_parquet(raw / "alerts.parquet")
    t0 = pd.Timestamp(config.TRAIN_START) - pd.Timedelta(days=2)   # 2 days of lag history

    a = alerts.assign(line=alerts["affected"].map(parse_lines), update_number=alerts["update_number"].astype(int))
    a["n_lines"] = a["line"].str.len()
    a = a.explode("line").dropna(subset=["line"])
    a = a.assign(cell=_cells(a["date"], t0), li=a["line"].map(LINE_TO_IDX).to_numpy())
    n_cells = int(a["cell"].max()) + 1
    shape = (n_cells, len(config.LINES))

    # one incident counts once per cell
    per_cell = a.loc[a["cell"] >= 0, ["cell", "li", "event_id"]].drop_duplicates()
    span = a.groupby(["event_id", "li"])["cell"].agg(["min", "max"]).reset_index()
    grids = {"any_alert": np.zeros(shape, np.int32), "new_delay": np.zeros(shape, np.int32)}
    np.add.at(grids["any_alert"], (per_cell["cell"].to_numpy(), per_cell["li"].to_numpy()), 1)
    first = span[span["min"] >= 0]
    np.add.at(grids["new_delay"], (first["min"].to_numpy(), first["li"].to_numpy()), 1)

    # in effect: +1 when an incident starts, -1 once it has been quiet for RECOVERY_MIN
    diff = np.zeros((n_cells + 1 + RECOVERY_MIN // STEP, len(config.LINES)), np.int32)
    lo = np.clip(span["min"].to_numpy(), 0, None)
    hi = np.clip(span["max"].to_numpy() + RECOVERY_MIN // STEP + 1, 0, None)
    np.add.at(diff, (lo, span["li"].to_numpy()), 1)
    np.add.at(diff, (hi, span["li"].to_numpy()), -1)
    grids["in_effect"] = (np.cumsum(diff, axis=0)[:n_cells] > 0).astype(np.int32)

    # the other (non-delay) alerts, used only as features
    d = pd.read_parquet(raw / "disruptions.parquet")
    d = d.assign(line=d["affected"].map(parse_lines), status=d["status_label"].str.split(r" \| "))
    d = d.explode("line").dropna(subset=["line"])
    d = d.assign(cell=_cells(d["date"], t0), li=d["line"].map(LINE_TO_IDX).to_numpy())
    d = d[(d["cell"] >= 0) & (d["cell"] < n_cells)]
    is_schedule = d["status"].map(lambda st: set(st) <= SCHEDULE_STATUSES)
    kinds = {"disrupt": d[~is_schedule], "schedule": d[is_schedule],
             "suspend": d[d["status_label"].str.contains("suspended")]}
    for name, part in kinds.items():
        part = part[["cell", "li", "event_id"]].drop_duplicates()
        grids[name] = np.zeros(shape, np.int32)
        np.add.at(grids[name], (part["cell"].to_numpy(), part["li"].to_numpy()), 1)

    cum, cum_sys, last = {}, {}, {}
    for name, C in grids.items():
        cum[name], cum_sys[name], last[name] = _accumulate(C)

    # every update, with what its text says, for the incident features
    a = a.sort_values(["li", "cell", "date", "update_number"])
    a["started"] = a.groupby(["event_id", "li"])["cell"].transform("min")
    a["n_update"] = a.groupby(["event_id", "li"]).cumcount() + 1
    a["cause"] = a["header"].map(cause_of).map({c: i for i, c in enumerate(CAUSE_NAMES)})
    a["phase"] = a["header"].map(phase_of)
    a["major"] = a["status_label"].str.contains(_MAJOR).astype(int)
    cols = ["cell", "started", "n_update", "cause", "phase", "major", "n_lines"]
    updates = [a.loc[a["li"] == i, cols].to_numpy() for i in range(len(config.LINES))]

    stations = pd.read_parquet(raw / "stations.parquet")
    profile = pd.read_parquet(raw / "ridership_profile.parquet")
    weather = weather_features(pd.read_parquet(raw / "weather.parquet"))
    day = weather.groupby(weather.index.date)["temperature_2m"]
    weather["temp_max_day"] = day.transform("max")        # heat waves: more signal & track failures
    weather["temp_min_day"] = day.transform("min")
    return Grid(t0, n_cells, cum, cum_sys, last, updates, line_overlap(stations),
                weather, line_ridership_profile(stations, profile))


def line_overlap(stations: pd.DataFrame) -> np.ndarray:
    routes = stations.assign(route=stations["daytime_routes"].str.split()).explode("route").reset_index(drop=True)
    routes["route"] = routes["route"].replace(config.LINE_ALIASES)
    routes = routes[routes["route"].isin(config.LINES)][["complex_id", "route"]].drop_duplicates()
    M = pd.crosstab(routes["complex_id"], routes["route"]).reindex(columns=config.LINES, fill_value=0)
    M = M.clip(upper=1).to_numpy().astype(np.float32)
    shared = M.T @ M
    W = shared / np.diag(shared)[:, None]
    np.fill_diagonal(W, 0)
    return W


def neighbor_features(g: Grid, j: np.ndarray, li: np.ndarray) -> dict:
    W = g.overlap[li]                                      # (rows, lines) weight of each other line
    out = {}
    for name, t, minutes in [("nbr_alerts_15m", "any_alert", 15), ("nbr_alerts_1h", "any_alert", 60),
                             ("nbr_alerts_3h", "any_alert", 180), ("nbr_new_1h", "new_delay", 60)]:
        recent = g.cum[t][j] - g.cum[t][np.maximum(j - minutes // STEP, 0)]
        out[name] = (W * recent).sum(axis=1)
    # the most-overlapping line that is (by our in-effect rule) still delayed right now
    active = (j[:, None] - g.last["any_alert"][j - 1]) * STEP <= RECOVERY_MIN
    out["nbr_top_active"] = (W * active).max(axis=1)
    return out


def incident_features(g: Grid, j: np.ndarray, li: np.ndarray) -> dict:
    """What the line's latest alert update (posted before cell j) says.

    Only counts if that update is less than INCIDENT_MEMORY old; otherwise the
    line has no current incident (inc_cause = "none").
    """
    n = len(j)
    out = {"inc_cause": np.zeros(n, np.int8), "inc_ongoing": np.full(n, -1, np.int8),
           "inc_major": np.zeros(n, np.int8), "inc_age_min": np.full(n, -1, np.int32),
           "inc_updates": np.zeros(n, np.int16), "inc_lines": np.zeros(n, np.int8)}
    for line in range(len(config.LINES)):
        rows = np.flatnonzero(li == line)
        u = g.updates[line]
        k = np.searchsorted(u[:, 0], j[rows], side="left") - 1        # latest update before cell j
        has = (k >= 0) & ((j[rows] - u[np.maximum(k, 0), 0]) * STEP <= INCIDENT_MEMORY)
        r, k = rows[has], k[has]
        out["inc_cause"][r] = u[k, 3]
        out["inc_ongoing"][r] = u[k, 4]
        out["inc_major"][r] = u[k, 5]
        out["inc_age_min"][r] = (j[r] - u[k, 1]) * STEP
        out["inc_updates"][r] = u[k, 2]
        out["inc_lines"][r] = u[k, 6]
    return out


# --- rows --------------------------------------------------------------------------
def make_rows(g: Grid, cells, lines, horizons, width=WINDOW) -> pd.DataFrame:
    """One row per (issue cell, line, horizon).

    The prediction is made at the start of `cells` (so everything in earlier
    cells is known) about the `width`-minute window starting `horizons`
    minutes later.
    """
    j = np.asarray(cells)
    li = np.asarray(lines)
    a = j + np.asarray(horizons) // STEP
    b = a + width // STEP
    target_ts = g.t0 + pd.to_timedelta(a * STEP, unit="min")

    out = {"issue_ts": g.t0 + pd.to_timedelta(j * STEP, unit="min"),
           "target_ts": target_ts, "horizon": np.asarray(horizons)}
    for t in TARGETS:
        out[f"label_{t}"] = (g.cum[t][b, li] - g.cum[t][a, li] > 0).astype(np.int8)

    kinds = {"alerts": "any_alert", "new": "new_delay"}
    for kind, t in kinds.items():
        for w, minutes in LAGS.items():
            lo = np.maximum(j - minutes // STEP, 0)
            out[f"line_{kind}_{w}"] = g.cum[t][j, li] - g.cum[t][lo, li]
            out[f"sys_{kind}_{w}"] = g.cum_sys[t][j] - g.cum_sys[t][lo]
    for kind, t in kinds.items():
        since = (j - g.last[t][j - 1, li]) * STEP
        out[f"min_since_line_{'alert' if kind == 'alerts' else 'new'}"] = np.minimum(since, 1440)
    out |= incident_features(g, j, li)
    out |= neighbor_features(g, j, li)
    for name, t, minutes in [("line_disrupt_1h", "disrupt", 60), ("line_disrupt_6h", "disrupt", 360),
                             ("line_suspend_1h", "suspend", 60), ("line_schedule_24h", "schedule", 1440)]:
        out[name] = g.cum[t][j, li] - g.cum[t][np.maximum(j - minutes // STEP, 0), li]
    out["sys_disrupt_1h"] = g.cum_sys["disrupt"][j] - g.cum_sys["disrupt"][np.maximum(j - 12, 0)]
    out["min_since_line_disrupt"] = np.minimum((j - g.last["disrupt"][j - 1, li]) * STEP, 1440)

    df = pd.DataFrame(out)
    df["inc_cause"] = pd.Categorical.from_codes(df["inc_cause"], categories=CAUSE_NAMES)
    ts = pd.DatetimeIndex(target_ts)
    df = pd.concat([df, calendar_features(ts)], axis=1)
    df["time_of_day"] = ts.hour + ts.minute / 60
    wx = g.weather.reindex(ts.floor("h")).reset_index(drop=True).fillna(0)
    df = pd.concat([df, wx], axis=1)

    df["line"] = pd.Categorical.from_codes(li, categories=config.LINES)
    riders = g.line_profile.set_index(["line", "dow", "hour"])["line_riders"]
    key = pd.MultiIndex.from_arrays([df["line"].astype(str), df["dow"], df["hour"]])
    df["line_riders"] = np.log1p(riders.reindex(key).fillna(0).to_numpy())
    return df


def sample_issues(g: Grid, start, end, n, rng, reach_min=None):
    """n random (issue cell, line) pairs whose windows all end before `end`."""
    reach = (reach_min or max(HORIZONS) + WINDOW) // STEP
    lo, hi = max(g.cell(start), 1), min(g.cell(end), g.n_cells) - reach
    return rng.integers(lo, hi, n), rng.integers(0, len(config.LINES), n)


# --- models ------------------------------------------------------------------------
def lgbm(**overrides):
    params = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)
    params.update(overrides)

    def make():
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    return make


# name -> (features, model factory, description)
MODELS = {
    "baseline": (["line", "dow", "hour"], lambda: HistoricalRate(["line", "dow", "hour"]),
                 "historical rate for this line x weekday x hour"),
    "logreg": (BASE, logreg(BASE), "logistic regression"),
    "hgb": (BASE, hgb(), "gradient boosting (sklearn, the current production model type)"),
    "lgbm": (BASE, lgbm(), "gradient boosting (LightGBM)"),
    "lgbm_text": (BASE + TEXT, lgbm(), "LightGBM + cause / phase / age of the current incident, read from the alert text"),
    "lgbm_nbr": (BASE + TEXT + NEIGHBORS, lgbm(), "... + recent delays on lines sharing stations"),
    "lgbm_context": (BASE + TEXT + NEIGHBORS + CONTEXT, lgbm(),
                     "... + suspensions / reroutes / schedule notices + daily temperature extremes"),
}
BEST = "lgbm_context"


def _samples(g: Grid, n_train: int, n_test: int):
    end = g.t0 + pd.Timedelta(minutes=STEP * g.n_cells)
    rng = np.random.default_rng(SEED)
    cells, lines = sample_issues(g, config.TRAIN_START, config.TEST_START, n_train, rng)
    train = make_rows(g, cells, lines, rng.choice(HORIZONS, n_train))
    # every test (moment, line) pair is scored at every horizon, so horizons are directly comparable
    rng = np.random.default_rng(SEED + 1)
    cells, lines = sample_issues(g, config.TEST_START, end, n_test, rng)
    rep = len(HORIZONS)
    test = make_rows(g, np.repeat(cells, rep), np.repeat(lines, rep), np.tile(HORIZONS, n_test))
    return train, test


def run(names=None, n_train=1_500_000, n_test=100_000, out_dir=OUT_DIR) -> pd.DataFrame:
    names = names or list(MODELS)
    t = time.time()
    g = load_grid()
    train, test = _samples(g, n_train, n_test)
    print(f"rows: {len(train):,} train, {len(test):,} test "
          f"({test['issue_ts'].min():%Y-%m-%d} to {test['issue_ts'].max():%Y-%m-%d}), {time.time() - t:.0f}s")

    results = []
    for target in TARGETS:
        y = train[f"label_{target}"]
        for name in names:
            feats, make, _ = MODELS[name]
            t = time.time()
            p = make().fit(train[feats], y).predict_proba(test[feats])[:, 1]
            print(f"  {target:<10} {name:<16} {time.time() - t:5.0f}s")
            for h in HORIZONS:
                m = (test["horizon"] == h).to_numpy()
                results.append({"target": target, "model": name, "horizon_min": h,
                                **score(test.loc[m, f"label_{target}"], p[m])})
    results = pd.DataFrame(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.csv"
    if path.exists():   # keep other models' rows; replace the ones we just re-ran
        old = pd.read_csv(path)
        results = pd.concat([old[~old["model"].isin(names)], results], ignore_index=True)
    results.to_csv(path, index=False)
    print(f"saved {path.relative_to(config.ROOT)}")
    return results


def summary(results: pd.DataFrame, metric="roc_auc") -> None:
    pd.set_option("display.width", 200)
    for target in TARGETS:
        r = results[results["target"] == target]
        if r.empty:
            continue
        rate = r.groupby("horizon_min")["delay_rate"].first()
        print(f"\n== {target}: {metric} by minutes ahead (happens in {rate.mean():.1%} of 15-min windows)")
        tbl = r.pivot(index="model", columns="horizon_min", values=metric)
        print(tbl.loc[tbl.mean(axis=1).sort_values(ascending=False).index].to_string())


def plot(results: pd.DataFrame, models=None, name="horizons_roc") -> None:
    import matplotlib.pyplot as plt
    from ml.plotting import GRAY, SERIES, TEXT_2, setup, show

    setup()
    models = models or [m for m in MODELS if m in set(results["model"])]
    others = [m for m in models if m != "baseline"]
    style = {m: (SERIES[i % len(SERIES)], "osD^vP*X"[i % 8]) for i, m in enumerate(others)}
    style["baseline"] = (GRAY, "s")
    labels = [f"{h}m" if h < 60 else f"{h // 60}h" for h in HORIZONS]
    targets = [t for t in TARGETS if t in set(results["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 4.8))
    for ax, target in zip(np.atleast_1d(axes), targets):
        r = results[results["target"] == target]
        ends = {}
        for m in models:
            s = r[r["model"] == m].set_index("horizon_min").reindex(HORIZONS)["roc_auc"]
            color, marker = style[m]
            ax.plot(range(len(HORIZONS)), s.values, color=color, marker=marker,
                    markersize=5, linewidth=2 if m != "baseline" else 1.5, label=m)
            ends[m] = s.values[-1]
        # end-of-line labels, nudged apart so converging lines stay readable
        span = np.subtract(*ax.get_ylim()[::-1])
        y_prev = -np.inf
        for m, y in sorted(ends.items(), key=lambda kv: kv[1]):
            y_prev = max(y, y_prev + span * 0.04)
            ax.text(len(HORIZONS) - 0.8, y_prev, m, color=TEXT_2, fontsize=8, va="center")
        ax.set_xticks(range(len(HORIZONS)), labels)
        ax.set_xlim(-0.3, len(HORIZONS) + 0.9)
        ax.set(xlabel="how far ahead", ylabel="roc_auc (0.5 = random)",
               title=f"{target} ({r['delay_rate'].mean():.1%} of 15-min windows)")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Accuracy by forecast horizon, held-out data from {config.TEST_START}",
                 x=0.01, ha="left", fontweight="bold")
    show(fig, name)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--plot" in sys.argv:
        res = pd.read_csv(OUT_DIR / "results.csv")
    elif "--quick" in sys.argv:
        res = run(args, 300_000, 20_000, out_dir=OUT_DIR / "quick")
    else:
        res = run(args)
    summary(res)
    plot(res)
