from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from app.extensions import db
from app.ml.predictor import artifacts_exist, predict_one
from app.models import Prediction

predict_bp = Blueprint("predict", __name__)

OCEAN_PROXIMITY_OPTIONS = [
    "NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND",
]

FIELD_GROUPS = [
    {
        "title": "Location",
        "note": "where the block group sits",
        "fields": [
            {"key": "latitude", "label": "Latitude", "default": 37.88,
             "min": 32.54, "max": 41.95, "step": 0.01, "unit": "\u00b0N", "scale": "linear",
             "help": "32.5 south \u2192 42.0 north"},
            {"key": "longitude", "label": "Longitude", "default": -122.23,
             "min": -124.35, "max": -114.31, "step": 0.01, "unit": "\u00b0W", "scale": "linear",
             "help": "-124.3 coast \u2192 -114.3 inland"},
        ],
    },
    {
        "title": "Structure",
        "note": "the building stock in this block group",
        "fields": [
            {"key": "housing_median_age", "label": "Median building age", "default": 28,
             "min": 1, "max": 52, "step": 1, "unit": "yrs", "scale": "linear", "help": None},
            {"key": "total_rooms", "label": "Total rooms", "default": 2635,
             "min": 2, "max": 39320, "step": 1, "unit": "rooms", "scale": "log",
             "help": "across the whole block group, not per house"},
            {"key": "total_bedrooms", "label": "Total bedrooms", "default": 538,
             "min": 1, "max": 6445, "step": 1, "unit": "bdrms", "scale": "log", "help": None},
        ],
    },
    {
        "title": "Community",
        "note": "who lives here",
        "fields": [
            {"key": "population", "label": "Population", "default": 1425,
             "min": 3, "max": 35682, "step": 1, "unit": "people", "scale": "log", "help": None},
            {"key": "households", "label": "Households", "default": 500,
             "min": 1, "max": 6082, "step": 1, "unit": "hh", "scale": "log", "help": None},
            {"key": "median_income", "label": "Median income", "default": 3.87,
             "min": 0.5, "max": 15.0, "step": 0.01, "unit": "\u00d7$10,000", "scale": "linear",
             "help": "3.87 \u2248 $38,700/yr"},
        ],
    },
]

# Flat list, derived once, used for parsing + defaults elsewhere.
FIELD_DEFS = [f for group in FIELD_GROUPS for f in group["fields"]]


def _parse_form(form):
    data = {}
    for field in FIELD_DEFS:
        data[field["key"]] = float(form.get(field["key"]))
    data["ocean_proximity"] = form.get("ocean_proximity", "INLAND")
    return data


@predict_bp.route("/predict", methods=["GET", "POST"])
def predict():
    model_ready = artifacts_exist(current_app)
    result = None
    submitted = None

    if request.method == "POST":
        if not model_ready:
            flash("Model artifacts not found. Run `python -m app.ml.train` first.", "error")
            return redirect(url_for("predict.predict"))
        try:
            features = _parse_form(request.form)
        except (TypeError, ValueError):
            flash("Please fill in every field with a valid number.", "error")
            return redirect(url_for("predict.predict"))

        if features["total_bedrooms"] > features["total_rooms"]:
            flash("Total bedrooms can't exceed total rooms for the block group.", "error")
            return redirect(url_for("predict.predict"))

        prediction_value = predict_one(current_app, features)

        record = Prediction(
            longitude=features["longitude"],
            latitude=features["latitude"],
            housing_median_age=features["housing_median_age"],
            total_rooms=features["total_rooms"],
            total_bedrooms=features["total_bedrooms"],
            population=features["population"],
            households=features["households"],
            median_income=features["median_income"],
            ocean_proximity=features["ocean_proximity"],
            predicted_value=prediction_value,
        )
        db.session.add(record)
        db.session.commit()

        result = prediction_value
        submitted = features

    return render_template(
        "predict.html",
        field_groups=FIELD_GROUPS,
        proximity_options=OCEAN_PROXIMITY_OPTIONS,
        model_ready=model_ready,
        result=result,
        submitted=submitted,
    )
