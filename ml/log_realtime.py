"""Log live subway train positions, so we can train on them later.

    python -m ml.log_realtime           # poll every minute until stopped (Ctrl+C)
    python -m ml.log_realtime --once    # one poll, to check it works

The MTA publishes where every train is (GTFS-realtime) but keeps no history,
so the only way to get training data is to record it ourselves. Gaps between
trains and trains falling behind schedule show up here minutes before a delay
alert is posted, which is the signal the alert-only model is missing.

Writes one parquet file per ~15 minutes to data/realtime/<date>/<HHMM>.parquet,
one row per train per poll (~40 MB a day).
"""
import sys
import time

import pandas as pd
import requests

from ml import config

FEEDS = ["gtfs", "gtfs-ace", "gtfs-bdfm", "gtfs-g", "gtfs-jz", "gtfs-nqrw", "gtfs-l"]
FEED_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2F{}"
OUT_DIR = config.DATA_DIR / "realtime"
POLL_SECONDS = 60
FLUSH_POLLS = 15


def poll() -> list[dict]:
    from google.transit import gtfs_realtime_pb2

    polled = pd.Timestamp.now(tz=config.TIMEZONE).tz_localize(None).floor("s")
    rows = {}
    for feed in FEEDS:
        try:
            r = requests.get(FEED_URL.format(feed), timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  {feed}: {e}")
            continue
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(r.content)
        for e in msg.entity:
            trip = e.trip_update.trip if e.HasField("trip_update") else e.vehicle.trip
            row = rows.setdefault(trip.trip_id, {
                "polled": polled, "feed_ts": msg.header.timestamp, "route": trip.route_id,
                "trip_id": trip.trip_id, "start_date": trip.start_date, "start_time": trip.start_time,
            })
            if e.HasField("vehicle"):
                v = e.vehicle
                row |= {"stop_id": v.stop_id, "status": v.current_status,
                        "stop_sequence": v.current_stop_sequence, "vehicle_ts": v.timestamp}
            if e.HasField("trip_update"):
                stops = e.trip_update.stop_time_update
                row["stops_left"] = len(stops)
                if stops:
                    first = stops[0]
                    row |= {"next_stop": first.stop_id,
                            "next_arrival": first.arrival.time or first.departure.time}
    return list(rows.values())


def flush(buffer: list[dict]) -> None:
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    first = df["polled"].min()
    path = OUT_DIR / f"{first:%Y-%m-%d}" / f"{first:%H%M}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  wrote {len(df):,} rows to {path.relative_to(config.ROOT)}")


def main(once: bool = False) -> None:
    buffer, polls = [], 0
    try:
        while True:
            t = time.time()
            rows = poll()
            buffer += rows
            polls += 1
            print(f"{pd.Timestamp.now():%H:%M:%S} {len(rows)} trains")
            if once or polls % FLUSH_POLLS == 0:
                flush(buffer)
                buffer = []
            if once:
                return
            time.sleep(max(0, POLL_SECONDS - (time.time() - t)))
    except KeyboardInterrupt:
        flush(buffer)


if __name__ == "__main__":
    main(once="--once" in sys.argv)
