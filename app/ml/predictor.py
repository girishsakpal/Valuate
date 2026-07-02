"""
Loads the trained pipeline + model once per process and exposes a
single `predict_one()` helper used by the Flask routes.
"""
import json
import os

import joblib
import pandas as pd

_model = None
_pipeline = None
_metrics_cache = None


def _paths(app):
    return app.config["MODEL_PATH"], app.config["PIPELINE_PATH"], app.config["METRICS_PATH"]


def artifacts_exist(app) -> bool:
    model_path, pipeline_path, _ = _paths(app)
    return os.path.exists(model_path) and os.path.exists(pipeline_path)


def load_artifacts(app):
    global _model, _pipeline
    model_path, pipeline_path, _ = _paths(app)
    _model = joblib.load(model_path)
    _pipeline = joblib.load(pipeline_path)
    return _model, _pipeline


def get_metrics(app):
    global _metrics_cache
    _, _, metrics_path = _paths(app)
    if not os.path.exists(metrics_path):
        return None
    # Metrics are small; re-read each call is fine and keeps insights fresh
    # after a retrain without restarting the server.
    with open(metrics_path) as f:
        _metrics_cache = json.load(f)
    return _metrics_cache


def predict_one(app, features: dict) -> float:
    """features must contain the 9 raw housing.csv input columns."""
    global _model, _pipeline
    if _model is None or _pipeline is None:
        load_artifacts(app)

    row = pd.DataFrame([features])
    prepared = _pipeline.transform(row)
    prediction = _model.predict(prepared)[0]
    return float(prediction)
