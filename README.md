# divhacks2026: NYC Subway Delay Predictor

Predicts the chance of a delay on each NYC subway line over the next 1–6 hours.

## Repo layout

```
ml/            model pipeline (data download, features, training, live prediction)
api/           FastAPI server the app calls
frontend/      the app UI (see frontend/README.md for the API contract)
data/          downloaded + processed data (gitignored, regenerate with fetch_data)
models/        trained model (gitignored) + metrics.json
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
python -m ml.fetch_data     # ~5–10 min, downloads to data/raw/
python -m ml.train          # trains + prints test metrics, saves models/delay_model.joblib
python -m ml.predict        # sanity check: live risk for every line
```

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
