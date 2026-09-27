"""Load raw data and turn it into (features, label) rows.

Shared by ml.train (production model) and ml.evaluate (experiments) so both
see exactly the same data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml import config
from ml.features import build_features, hourly_delay_counts, line_ridership_profile

SEED = 42


@dataclass
class Data:
    counts: pd.DataFrame        # hourly delay-incident counts, one column per line
    weather: pd.DataFrame
    line_profile: pd.DataFrame  # typical ridership per (line, dow, hour)

    @property
    def end(self) -> pd.Timestamp:
        return self.counts.index.max()


def load_data() -> Data:
    raw = config.RAW_DIR
    alerts = pd.read_parquet(raw / "alerts.parquet")
    weather = pd.read_parquet(raw / "weather.parquet")
    stations = pd.read_parquet(raw / "stations.parquet")
    profile = pd.read_parquet(raw / "ridership_profile.parquet")
    counts = hourly_delay_counts(alerts, start=config.TRAIN_START, end=alerts["date"].max())
    return Data(counts, weather, line_ridership_profile(stations, profile))


def _grid(data: Data, start, end) -> pd.DataFrame:
    """Every (hour, line) in [start, end), with its 0/1 label."""
    idx = data.counts.index
    # skip the first 2 days so lag features have history to look back on
    lo = max(pd.Timestamp(start), idx[0] + pd.Timedelta(days=2))
    hours = idx[(idx >= lo) & (idx < (pd.Timestamp(end) if end else idx[-1] + pd.Timedelta(hours=1)))]
    rows = pd.DataFrame({
        "target_ts": np.repeat(hours, len(config.LINES)),
        "line": np.tile(config.LINES, len(hours)),
    })
    rows["label"] = (data.counts.loc[hours].stack().to_numpy() > 0).astype(int)
    return rows


def training_rows(data: Data, start=config.TRAIN_START, end=None) -> pd.DataFrame:
    """One row per (hour, line), each with one randomly sampled horizon.

    Sampling keeps the training set small while still teaching the model
    every horizon.
    """
    rows = _grid(data, start, end)
    rng = np.random.default_rng(SEED)
    rows["horizon"] = rng.integers(1, config.MAX_HORIZON_HOURS + 1, len(rows))
    return build_features(rows, data.counts, data.weather, data.line_profile)


def evaluation_rows(data: Data, start, end=None) -> pd.DataFrame:
    """One row per (hour, line, horizon) for *every* horizon 1..MAX.

    Evaluation scores every horizon so per-horizon metrics aren't noisy.
    """
    rows = _grid(data, start, end)
    horizons = np.arange(1, config.MAX_HORIZON_HOURS + 1)
    rows = rows.loc[rows.index.repeat(len(horizons))].reset_index(drop=True)
    rows["horizon"] = np.tile(horizons, len(rows) // len(horizons))
    return build_features(rows, data.counts, data.weather, data.line_profile)
