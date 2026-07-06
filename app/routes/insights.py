import json

from flask import Blueprint, current_app, render_template

from app.ml.predictor import get_metrics

insights_bp = Blueprint("insights", __name__)


@insights_bp.route("/insights")
def insights():
    metrics = get_metrics(current_app)
    metrics_json = json.dumps(metrics) if metrics else "null"
    return render_template("insights.html", metrics=metrics, metrics_json=metrics_json)
