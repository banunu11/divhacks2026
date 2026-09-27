"""Live predictions.

    python -m ml.predict            # prints next-6h risk for every line
    python -m ml.predict A 4 L      # just these lines

Recent delay history comes from MTA's live GTFS-RT alerts feed (JSON), and
weather from Open-Meteo's forecast API. Every live fetch is appended to
data/live_alerts_log.parquet so lag features get richer the longer the API runs.
"""
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests

from ml import config
from ml.features import build_features, hourly_delay_counts

TZ = ZoneInfo(config.TIMEZONE)
LIVE_LOG = config.DATA_DIR / "live_alerts_log.parquet"


def now_local() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(TZ).replace(tzinfo=None))


def _to_local(epoch: int) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromtimestamp(epoch, TZ).replace(tzinfo=None))


def fetch_live_alerts() -> pd.DataFrame:
    """Current subway delay alerts, in the same shape as the historical data."""
    r = requests.get(config.LIVE_ALERTS_URL, timeout=20)
    r.raise_for_status()
    rows = []
    for ent in r.json().get("entity", []):
        alert = ent.get("alert", {})
        meta = alert.get("transit_realtime.mercury_alert", {})
        if "delay" not in str(meta.get("alert_type", "")).lower():
            continue
        routes = [e["route_id"] for e in alert.get("informed_entity", []) if "route_id" in e]
        header = next((t["text"] for t in alert.get("header_text", {}).get("translation", [])
                       if t.get("language") == "en"), "")
        for key in ("created_at", "updated_at"):
            if meta.get(key):
                rows.append({
                    "event_id": ent.get("id"),
                    "date": _to_local(meta[key]),
                    "affected": " | ".join(routes),
                    "header": header,
                })
    return pd.DataFrame(rows, columns=["event_id", "date", "affected", "header"])


def recent_alerts() -> pd.DataFrame:
    """Live feed merged with everything we've logged from it before."""
    live = fetch_live_alerts()
    if LIVE_LOG.exists():
        live = pd.concat([pd.read_parquet(LIVE_LOG), live], ignore_index=True)
    live = live.drop_duplicates(["event_id", "date"])
    live = live[live["date"] >= now_local() - pd.Timedelta(days=2)]
    live.to_parquet(LIVE_LOG, index=False)
    return live


def fetch_weather_forecast() -> pd.DataFrame:
    r = requests.get(config.WEATHER_FORECAST_URL, params={
        "latitude": config.NYC_LAT,
        "longitude": config.NYC_LON,
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": config.TIMEZONE,
        "past_days": 1,
        "forecast_days": 2,
    }, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "hour_ts"})
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    return df


class DelayPredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.line_profile = bundle["line_profile"]
        self.metrics = bundle.get("metrics", {})
        self.trained_through = bundle.get("trained_through")

    def predict(self, lines=None, hours_ahead: int = config.MAX_HORIZON_HOURS) -> dict:
        lines = [l.upper() for l in (lines or config.LINES) if l.upper() in config.LINES]
        hours_ahead = max(1, min(hours_ahead, config.MAX_HORIZON_HOURS))
        now = now_local().floor("h")

        alerts = recent_alerts()
        counts = hourly_delay_counts(alerts, start=now - pd.Timedelta(hours=48), end=now)
        weather = fetch_weather_forecast()

        rows = pd.DataFrame([
            {"target_ts": now + pd.Timedelta(hours=h), "line": line, "horizon": h}
            for line in lines for h in range(1, hours_ahead + 1)
        ])
        df = build_features(rows, counts, weather, self.line_profile)
        df["probability"] = self.model.predict_proba(df[self.features])[:, 1]

        active = {line: [] for line in lines}
        for _, a in alerts[alerts["date"] >= now - pd.Timedelta(hours=2)].iterrows():
            for line in str(a["affected"]).split(" | "):
                if line in active and a["header"] not in active[line]:
                    active[line].append(a["header"])

        return {
            "generated_at": now_local().isoformat(),
            "lines": [
                {
                    "line": line,
                    "active_alerts": active[line],
                    "forecast": [
                        {
                            "hour": r.target_ts.isoformat(),
                            "hours_ahead": int(r.horizon),
                            "delay_probability": round(float(r.probability), 3),
                            "risk": risk_bucket(r.probability),
                        }
                        for r in g.sort_values("horizon").itertuples()
                    ],
                }
                for line, g in df.groupby("line", observed=True, sort=False)
            ],
        }


def risk_bucket(p: float) -> str:
    return "high" if p >= config.RISK_HIGH else "medium" if p >= config.RISK_MEDIUM else "low"


if __name__ == "__main__":
    result = DelayPredictor().predict(sys.argv[1:] or None)
    print(f"generated {result['generated_at']}")
    for entry in result["lines"]:
        probs = "  ".join(f"+{f['hours_ahead']}h {f['delay_probability']:.0%}" for f in entry["forecast"])
        flag = "  <- ACTIVE: " + entry["active_alerts"][0][:60] if entry["active_alerts"] else ""
        print(f"  {entry['line']:>2}: {probs}{flag}")
