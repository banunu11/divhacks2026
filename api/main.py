"""HTTP API for the frontend.

    uvicorn api.main:app --reload --port 8000

Interactive docs at http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ml import config
from ml.predict import DelayPredictor

app = FastAPI(title="NYC Subway Delay Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon: let any frontend dev server call us
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: DelayPredictor | None = None


def predictor() -> DelayPredictor:
    global _predictor
    if _predictor is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run: python -m ml.train")
        _predictor = DelayPredictor()
    return _predictor


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": config.MODEL_PATH.exists()}


@app.get("/lines")
def lines():
    return {"lines": config.LINES}


@app.get("/predict")
def predict(
    lines: str | None = Query(None, description="Comma-separated lines, e.g. 'A,4,L'. Omit for all."),
    hours: int = Query(config.MAX_HORIZON_HOURS, ge=1, le=config.MAX_HORIZON_HOURS),
):
    wanted = [l.strip() for l in lines.split(",")] if lines else None
    return predictor().predict(wanted, hours)


@app.get("/predict/{line}")
def predict_line(line: str, hours: int = Query(config.MAX_HORIZON_HOURS, ge=1, le=config.MAX_HORIZON_HOURS)):
    if line.upper() not in config.LINES:
        raise HTTPException(404, f"Unknown line '{line}'. See /lines.")
    return predictor().predict([line], hours)["lines"][0]


@app.get("/model")
def model_info():
    p = predictor()
    return {"trained_through": p.trained_through, "metrics": p.metrics}
