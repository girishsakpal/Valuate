from flask import Blueprint, current_app, jsonify, request

from app.extensions import db
from app.ml.predictor import artifacts_exist, get_metrics, predict_one
from app.models import Prediction

api_bp = Blueprint("api", __name__)

REQUIRED_FIELDS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity",
]


@api_bp.route("/predict", methods=["POST"])
def api_predict():
    if not artifacts_exist(current_app):
        return jsonify({"error": "Model not trained yet. Run `python -m app.ml.train`."}), 503

    payload = request.get_json(silent=True) or {}
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        features = {k: (float(payload[k]) if k != "ocean_proximity" else payload[k]) for k in REQUIRED_FIELDS}
    except (TypeError, ValueError):
        return jsonify({"error": "Numeric fields must be numbers."}), 400

    prediction = predict_one(current_app, features)

    if payload.get("save", True):
        record = Prediction(predicted_value=prediction, **features)
        db.session.add(record)
        db.session.commit()

    return jsonify({"predicted_value": round(prediction, 2)})


@api_bp.route("/metrics", methods=["GET"])
def api_metrics():
    metrics = get_metrics(current_app)
    if metrics is None:
        return jsonify({"error": "No metrics available. Train the model first."}), 404
    return jsonify(metrics)


@api_bp.route("/predictions", methods=["GET"])
def api_predictions():
    limit = request.args.get("limit", 50, type=int)
    records = (
        Prediction.query.order_by(Prediction.created_at.desc()).limit(min(limit, 200)).all()
    )
    return jsonify([r.to_dict() for r in records])
