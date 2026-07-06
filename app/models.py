from datetime import datetime, timezone
from app.extensions import db


class Prediction(db.Model):

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Inputs (mirrors the housing.csv schema)
    longitude = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    housing_median_age = db.Column(db.Float, nullable=False)
    total_rooms = db.Column(db.Float, nullable=False)
    total_bedrooms = db.Column(db.Float, nullable=False)
    population = db.Column(db.Float, nullable=False)
    households = db.Column(db.Float, nullable=False)
    median_income = db.Column(db.Float, nullable=False)
    ocean_proximity = db.Column(db.String(32), nullable=False)

    # Output
    predicted_value = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(64), default="rf_v1")

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "housing_median_age": self.housing_median_age,
            "total_rooms": self.total_rooms,
            "total_bedrooms": self.total_bedrooms,
            "population": self.population,
            "households": self.households,
            "median_income": self.median_income,
            "ocean_proximity": self.ocean_proximity,
            "predicted_value": self.predicted_value,
            "model_version": self.model_version,
        }
