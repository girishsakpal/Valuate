import os

from flask import Flask

from app.config import config_by_name
from app.extensions import db


def create_app(env=None):
    env = env or os.environ.get("APP_ENV", os.environ.get("FLASK_ENV", "development"))
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_by_name.get(env, config_by_name["development"]))

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)

    db.init_app(app)

    with app.app_context():
        from app import models  # noqa: F401  (register models)
        db.create_all()

    from app.routes.main import main_bp
    from app.routes.predict import predict_bp
    from app.routes.history import history_bp
    from app.routes.insights import insights_bp
    from app.routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(insights_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.context_processor
    def inject_globals():
        from datetime import datetime
        return {"current_year": datetime.utcnow().year}

    @app.errorhandler(404)
    def not_found(e):
        from flask import render_template
        return render_template("404.html"), 404

    return app
