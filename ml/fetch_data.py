"""Download raw data into data/raw/.

    python -m ml.fetch_data            # everything
    python -m ml.fetch_data alerts     # just one source

Sources:
  alerts     MTA Service Alerts (subway, delay-type only)  -> our labels
  disruptions  the other subway alerts (suspensions, reroutes, ...) -> features
  stations   MTA Subway Stations                           -> station -> route map
  ridership  MTA Subway Hourly Ridership (aggregated server-side to a
             station x day-of-week x hour profile, so we don't pull 120M rows)
  weather    Open-Meteo hourly archive for NYC
"""
import os
import sys
import time

import pandas as pd
import requests

from ml import config

APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")  # optional, raises rate limits
PAGE = 50_000


def socrata(dataset_id: str, params: dict, paginate: bool = True) -> pd.DataFrame:
    """Query a data.ny.gov dataset with SoQL, following pagination."""
    url = f"{config.SOCRATA}/{dataset_id}.json"
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    frames, offset = [], 0
    while True:
        q = {**params, "$limit": PAGE, "$offset": offset}
        for attempt in range(5):
            try:
                r = requests.get(url, params=q, headers=headers, timeout=300)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 4:
                    raise
                print(f"  retry {attempt + 1} after error: {e}")
                time.sleep(2 ** attempt)
        rows = r.json()
        frames.append(pd.DataFrame(rows))
        print(f"  {dataset_id}: fetched {offset + len(rows):,} rows")
        if not paginate or len(rows) < PAGE:
            break
        offset += PAGE
    return pd.concat(frames, ignore_index=True)


def fetch_alerts() -> None:
    df = socrata(config.ALERTS_ID, {
        "$select": "event_id, update_number, date, status_label, affected, header",
        "$where": "agency = 'NYCT Subway' AND status_label like '%delays%'",
        "$order": "date",
    })
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(config.RAW_DIR / "alerts.parquet", index=False)
    print(f"alerts: {len(df):,} rows, {df['date'].min()} -> {df['date'].max()}")


def fetch_disruptions() -> None:
    """Every other subway alert: suspensions, skipped stops, reroutes, schedule notices."""
    df = socrata(config.ALERTS_ID, {
        "$select": "event_id, update_number, date, status_label, affected, header",
        "$where": "agency = 'NYCT Subway' AND status_label not like '%delays%'",
        "$order": "date",
    })
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(config.RAW_DIR / "disruptions.parquet", index=False)
    print(f"disruptions: {len(df):,} rows, {df['date'].min()} -> {df['date'].max()}")


def fetch_stations() -> None:
    df = socrata(config.STATIONS_ID, {
        "$select": "complex_id, stop_name, borough, line, daytime_routes",
    })
    df.to_parquet(config.RAW_DIR / "stations.parquet", index=False)
    print(f"stations: {len(df):,} rows")


def fetch_ridership_profile(start: str = "2024-09-09", weeks: int = 8) -> None:
    """Average hourly entries per station complex for each (day-of-week, hour).

    Aggregating a whole year server-side (date_extract_*) times out on
    Socrata, but grouping one day at a time by the raw timestamp takes ~1s, so
    we pull a stretch of normal fall weeks day by day and average locally.
    """
    days = []
    for day in pd.date_range(start, periods=weeks * 7, freq="D"):
        nxt = day + pd.Timedelta(days=1)
        days.append(socrata(config.RIDERSHIP_ID, {
            "$select": "station_complex_id, transit_timestamp, sum(ridership) AS riders",
            "$where": (
                f"transit_mode = 'subway' AND transit_timestamp >= '{day:%Y-%m-%dT%H:%M:%S}' "
                f"AND transit_timestamp < '{nxt:%Y-%m-%dT%H:%M:%S}'"
            ),
            "$group": "station_complex_id, transit_timestamp",
        }, paginate=False))
    df = pd.concat(days, ignore_index=True)
    ts = pd.to_datetime(df["transit_timestamp"])
    df = (
        df.assign(dow=ts.dt.dayofweek, hour=ts.dt.hour, riders=df["riders"].astype(float))
        .groupby(["station_complex_id", "dow", "hour"], as_index=False)["riders"].sum()
    )
    df["riders"] /= weeks  # -> average entries in that hour on that weekday
    df.to_parquet(config.RAW_DIR / "ridership_profile.parquet", index=False)
    print(f"ridership profile: {len(df):,} rows")


def fetch_weather(start: str = "2020-04-01", end: str | None = None) -> None:
    end = end or (pd.Timestamp.now() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    r = requests.get(config.WEATHER_ARCHIVE_URL, params={
        "latitude": config.NYC_LAT,
        "longitude": config.NYC_LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": config.TIMEZONE,
    }, timeout=300)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "hour_ts"})
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    df.to_parquet(config.RAW_DIR / "weather.parquet", index=False)
    print(f"weather: {len(df):,} hours")


FETCHERS = {
    "alerts": fetch_alerts,
    "disruptions": fetch_disruptions,
    "stations": fetch_stations,
    "ridership": fetch_ridership_profile,
    "weather": fetch_weather,
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(FETCHERS)
    for name in targets:
        print(f"== {name}")
        FETCHERS[name]()
