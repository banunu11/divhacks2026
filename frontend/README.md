# Frontend

UI team's folder: put the app here (React, Next, Expo, whatever you like).

## Talking to the model

Start the backend from the repo root (see the main README), then call:

| Endpoint | What it returns |
|---|---|
| `GET /lines` | `{"lines": ["1","2",...,"Z"]}` |
| `GET /predict` | Forecast for **all** lines, next 6 hours |
| `GET /predict?lines=A,4,L&hours=3` | Only those lines, next 3 hours |
| `GET /predict/A` | One line |
| `GET /model` | Model accuracy metrics (good for a "how it works" screen) |
| `GET /health` | `{"ok": true, "model_loaded": true}` |

Base URL for local dev: `http://localhost:8000`. CORS is open, so you can call it straight from a browser dev server.
Live, clickable docs: http://localhost:8000/docs

### Response shape (`GET /predict/A`)

```json
{
  "line": "A",
  "active_alerts": ["Southbound [A] trains are delayed while we address a signal problem near 59 St"],
  "forecast": [
    {"hour": "2026-09-26T23:00:00", "hours_ahead": 1, "delay_probability": 0.42, "risk": "medium"},
    {"hour": "2026-09-27T00:00:00", "hours_ahead": 2, "delay_probability": 0.31, "risk": "medium"}
  ]
}
```

- `delay_probability`: 0–1 chance MTA posts a delay alert for that line during that hour.
- `risk`: `"low" | "medium" | "high"`, a pre-bucketed value you can map to colors.
- `active_alerts`: what MTA is reporting right now (last 2h), so you can show it next to the prediction.
- `hour`: NYC local time, start of the hour.

The `/predict` (all lines) response is `{"generated_at": "...", "lines": [ <the object above>, ... ]}`.

Official MTA line colors if you want them: 1/2/3 `#EE352E`, 4/5/6 `#00933C`, 7 `#B933AD`, A/C/E `#0039A6`,
B/D/F/M `#FF6319`, G `#6CBE45`, J/Z `#996633`, L `#A7A9AC`, N/Q/R/W `#FCCC0A`.
