# divhacks2026: NYC Subway Delay Predictor

Predicts the chance of a delay on each NYC subway line over the next 1–6 hours.

## Repo layout

```
ml/            model pipeline (data download, features, training, evaluation, live prediction)
api/           FastAPI server the app calls
frontend/      the app UI (see frontend/README.md for the API contract)
notebooks/     data_analysis.ipynb (explore the data) + evaluate.ipynb (compare models), both with charts
data/          downloaded + processed data (gitignored, regenerate with fetch_data)
models/        production model + experiments/<name>/ (one folder per model; metrics committed, weights gitignored)
reports/       saved charts
```

## Setup (one time)

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Build the model

```bash
python -m ml.fetch_data     # ~2 min, downloads to data/raw/
python -m ml.evaluate run all   # optional: score every model on held-out data
python -m ml.train          # trains the production model -> models/delay_model.joblib
python -m ml.predict        # sanity check: live risk for every line
```

## Compare models

Models are defined in `ml/experiments.py`. Each one is trained on data before
Aug 2025 and tested on everything after, which is data it never saw.

```bash
python -m ml.evaluate list                # what's in the registry
python -m ml.evaluate run all             # train + test every model (~1.5 min)
python -m ml.evaluate compare             # leaderboard
python -m ml.evaluate show hgb_full       # detailed breakdown
python -m ml.evaluate replay hgb_full 2026-07-14   # predictions vs reality for one day
python -m ml.train hgb_full               # promote a model to production (the API uses it)
```

With charts: open `notebooks/evaluate.ipynb` or `notebooks/data_analysis.ipynb` and select the `.venv` kernel.

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000/docs. The endpoints and response format are in [frontend/README.md](frontend/README.md).

## How the model works

**Question it answers:** will MTA post a delay alert for line *L* during hour *t*, using only what we know *h* hours before (h = 1–6)?

| Data | Source | Used for |
|---|---|---|
| Service alerts (2020→now) | [data.ny.gov `7kct-peq7`](https://data.ny.gov/d/7kct-peq7) | **Labels** + recent-delay features |
| Hourly ridership (2020–2024) | [data.ny.gov `wujg-7c2s`](https://data.ny.gov/d/wujg-7c2s) | Typical crowding per line / day-of-week / hour |
| Subway stations | [data.ny.gov `39hk-dx4f`](https://data.ny.gov/d/39hk-dx4f) | Maps station ridership onto lines |
| Weather | [Open-Meteo](https://open-meteo.com) archive + forecast | Rain, snow, temperature, wind |
| Live alerts | MTA GTFS-RT alerts feed (JSON) | Recent delays at prediction time |

Note: the ridership dataset has no delay information, so by itself it can't train a delay model. The delay labels come from the Service Alerts dataset, and ridership is a crowding feature.

**Features:** line, hours ahead, hour, day-of-week, month, weekend/holiday/rush-hour flags, weather at the target hour, typical line ridership, delays on this line in the last 1/3/6/24h, systemwide delays in the last 1/3/6/24h, hours since this line's last delay.

**Model:** scikit-learn `HistGradientBoostingClassifier`. Validated on a time-based holdout (the most recent year), compared against a baseline of "historical delay rate for this line at this hour of the week." Results are in `models/metrics.json`.

## Ideas if we have time

- Planned-work alerts as a feature (they're announced days ahead)
- Station-level predictions (which part of the line)
- Big-event calendar (Yankees/Mets/MSG games, parades)
- Retrain on a cron as new alert data lands
