"""
Preprocessing pipeline shared between training and inference.

Numerical columns: median-impute missing values, then standard-scale.
Categorical column (ocean_proximity): one-hot encode.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_ATTRIBS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CAT_ATTRIBS = ["ocean_proximity"]


def build_pipeline(num_attribs=None, cat_attribs=None) -> ColumnTransformer:
    num_attribs = num_attribs or NUM_ATTRIBS
    cat_attribs = cat_attribs or CAT_ATTRIBS

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline
