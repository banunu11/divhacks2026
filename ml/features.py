"""Feature engineering shared by training (ml.train) and serving (ml.predict).

Unit of prediction: (line, target hour, horizon). "Will there be a delay alert
on line L during hour t, given what we knew h hours earlier?"

All timestamps are naive NYC local time, floored to the hour.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from ml import config

LINE_TO_IDX = {line: i for i, line in enumerate(config.LINES)}

FEATURES = (
    ["line", "horizon", "hour", "dow", "month", "is_weekend", "is_holiday", "is_rush"]
    + config.WEATHER_VARS
    + ["precip_3h", "snow_24h", "line_riders"]
    + [f"line_delays_{w}h" for w in config.LAG_WINDOWS_HOURS]
    + [f"sys_delays_{w}h" for w in config.LAG_WINDOWS_HOURS]
    + ["hours_since_line_delay"]
)
CATEGORICAL = ["line"]


# --- alerts -> hourly delay counts ---------------------------------------
def parse_lines(affected: str) -> list[str]:
    """'A | C | E' -> ['A', 'C', 'E'], dropping lines we don't model."""
    out = []
    for tok in str(affected).split("|"):
        tok = tok.strip().upper()
        tok = config.LINE_ALIASES.get(tok, tok)
        if tok in LINE_TO_IDX and tok not in out:
            out.append(tok)
    return out


def hourly_delay_counts(alerts: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Number of distinct delay incidents touching each line, per hour.

    `alerts` needs columns: event_id, date (naive local), affected.
    Returns a DataFrame indexed by a complete hourly range, one column per line.
    """
    a = alerts[["event_id", "date", "affected"]].copy()
    a["hour_ts"] = pd.to_datetime(a["date"]).dt.floor("h")
    a["line"] = a["affected"].map(parse_lines)
    a = a.explode("line").dropna(subset=["line"])
    counts = (
        a.drop_duplicates(["event_id", "hour_ts", "line"])
        .groupby(["hour_ts", "line"]).size()
        .unstack(fill_value=0)
        .reindex(columns=config.LINES, fill_value=0)
    )
    start = pd.Timestamp(start) if start is not None else counts.index.min()
    end = pd.Timestamp(end) if end is not None else counts.index.max()
    full = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
    return counts.reindex(full, fill_value=0).astype(np.int32)


# --- ridership -> per-line crowding profile -------------------------------
def line_ridership_profile(stations: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Typical hourly entries at stations served by each line, by (dow, hour).

    A station's full ridership counts toward every line serving it — a crowded
    Times Sq platform affects all of its lines.
    """
    routes = (
        stations.assign(route=stations["daytime_routes"].str.split())
        .explode("route")[["complex_id", "route"]]
    )
    routes["route"] = routes["route"].replace(config.LINE_ALIASES)
    routes = routes[routes["route"].isin(config.LINES)].drop_duplicates()
    p = profile.rename(columns={"station_complex_id": "complex_id"})
    merged = p.merge(routes, on="complex_id")
    out = (
        merged.groupby(["route", "dow", "hour"])["riders"].sum()
        .rename("line_riders").reset_index().rename(columns={"route": "line"})
    )
    return out


# --- weather -----------------------------------------------------------------
def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.sort_values("hour_ts").set_index("hour_ts")[config.WEATHER_VARS].copy()
    w = w.ffill().fillna(0)
    w["precip_3h"] = w["precipitation"].rolling(3, min_periods=1).sum()
    w["snow_24h"] = w["snowfall"].rolling(24, min_periods=1).sum()
    return w


# --- lag features ----------------------------------------------------------
def lag_features(counts: pd.DataFrame, issue_ts: pd.Series, lines: pd.Series) -> pd.DataFrame:
    """Delay history as of `issue_ts` (inclusive of that hour).

    Vectorized via cumulative sums so it's fast on ~1M rows.
    """
    C = counts.to_numpy()
    S = np.vstack([np.zeros((1, C.shape[1])), np.cumsum(C, axis=0)])
    sys_S = S.sum(axis=1)
    t0 = counts.index[0]

    j = ((pd.DatetimeIndex(issue_ts) - t0) // pd.Timedelta(hours=1)).to_numpy()
    j = np.clip(j, -1, len(C) - 1)
    li = lines.map(LINE_TO_IDX).to_numpy()
    out = {}
    for w in config.LAG_WINDOWS_HOURS:
        hi, lo = j + 1, np.maximum(j + 1 - w, 0)
        out[f"line_delays_{w}h"] = S[hi, li] - S[lo, li]
        out[f"sys_delays_{w}h"] = sys_S[hi] - sys_S[lo]

    # hours since the line last had a delay (capped at 24 = "not recently")
    idx = np.arange(len(C))[:, None]
    last = np.where(C > 0, idx, -10_000)
    last = np.maximum.accumulate(last, axis=0)
    since = j - last[np.clip(j, 0, None), li]
    out["hours_since_line_delay"] = np.clip(since, 0, 24)
    return pd.DataFrame(out, index=issue_ts.index)


# --- calendar ---------------------------------------------------------------
_HOLIDAYS = USFederalHolidayCalendar().holidays("2019-01-01", "2030-12-31")


def calendar_features(ts: pd.Series) -> pd.DataFrame:
    ts = pd.DatetimeIndex(ts)
    hour, dow = ts.hour, ts.dayofweek
    weekday = dow < 5
    return pd.DataFrame({
        "hour": hour,
        "dow": dow,
        "month": ts.month,
        "is_weekend": (~weekday).astype(int),
        "is_holiday": ts.normalize().isin(_HOLIDAYS).astype(int),
        "is_rush": (weekday & (((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19)))).astype(int),
    })


# --- assemble ---------------------------------------------------------------
def build_features(
    rows: pd.DataFrame,
    counts: pd.DataFrame,
    weather: pd.DataFrame,
    line_profile: pd.DataFrame,
) -> pd.DataFrame:
    """rows: DataFrame with columns target_ts, line, horizon (hours ahead, >=1).

    Returns rows with all FEATURES columns attached.
    """
    rows = rows.reset_index(drop=True)
    issue_ts = rows["target_ts"] - pd.to_timedelta(rows["horizon"], unit="h")

    cal = calendar_features(rows["target_ts"])
    wx = weather_features(weather).reindex(pd.DatetimeIndex(rows["target_ts"]))
    wx = wx.reset_index(drop=True).fillna(0)
    lags = lag_features(counts, issue_ts, rows["line"])

    out = pd.concat([rows, cal, wx, lags], axis=1)
    out = out.merge(line_profile, on=["line", "dow", "hour"], how="left")
    out["line_riders"] = np.log1p(out["line_riders"].fillna(0))
    out["line"] = pd.Categorical(out["line"], categories=config.LINES)
    return out
