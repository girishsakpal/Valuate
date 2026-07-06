import json

import pandas as pd
from flask import Blueprint, current_app, render_template

from app.extensions import db
from app.models import Prediction

main_bp = Blueprint("main", __name__)

_dataset_cache = None


def _load_dataset(app):
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = pd.read_csv(app.config["DATA_PATH"])
    return _dataset_cache


@main_bp.route("/")
def dashboard():
    df = _load_dataset(current_app)

    summary = {
        "rows": int(len(df)),
        "median_price": float(df["median_house_value"].median()),
        "mean_price": float(df["median_house_value"].mean()),
        "median_income": float(df["median_income"].median()),
        "avg_rooms": float(df["total_rooms"].mean()),
        "proximity_counts": df["ocean_proximity"].value_counts().to_dict(),
    }

    # Map scatter: lat/long colored by price (sampled for payload size)
    map_sample = df.sample(n=min(2500, len(df)), random_state=42)
    map_data = {
        "lon": map_sample["longitude"].tolist(),
        "lat": map_sample["latitude"].tolist(),
        "price": map_sample["median_house_value"].tolist(),
        "population": map_sample["population"].fillna(0).tolist(),
    }

    # Price distribution histogram bins (precomputed server-side)
    price_hist = df["median_house_value"].tolist()

    # Correlation matrix for a heatmap
    numeric_cols = [
        "median_house_value", "median_income", "housing_median_age",
        "total_rooms", "total_bedrooms", "population", "households",
        "latitude", "longitude",
    ]
    corr = df[numeric_cols].corr().round(2)
    correlation = {
        "columns": numeric_cols,
        "matrix": corr.values.tolist(),
    }

    recent_predictions = (
        Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()
    )
    total_predictions = db.session.query(Prediction).count()

    return render_template(
        "dashboard.html",
        summary=summary,
        map_data=json.dumps(map_data),
        price_hist=json.dumps(price_hist),
        correlation=json.dumps(correlation),
        recent_predictions=recent_predictions,
        total_predictions=total_predictions,
    )
