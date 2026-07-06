import os

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")
    MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
    PIPELINE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
    METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
    DATA_PATH = os.path.join(BASE_DIR, "data", "housing.csv")
    RECORDS_PER_PAGE = 12


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'housing_dev.db')}"
    )


class ProductionConfig(BaseConfig):
    DEBUG = False
    _prod_url = os.environ.get("DATABASE_URL")
    if _prod_url and _prod_url.startswith("postgres://"):
        _prod_url = _prod_url.replace("postgres://", "postgresql://", 1)
    SQLALCHEMY_DATABASE_URI = _prod_url or "sqlite:///housing_prod_fallback.db"


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
