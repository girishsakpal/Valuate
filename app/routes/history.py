from flask import Blueprint, current_app, render_template, request

from app.models import Prediction

history_bp = Blueprint("history", __name__)


@history_bp.route("/history")
def history():
    page = request.args.get("page", 1, type=int)
    per_page = current_app.config["RECORDS_PER_PAGE"]
    pagination = Prediction.query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return render_template("history.html", pagination=pagination, records=pagination.items)
