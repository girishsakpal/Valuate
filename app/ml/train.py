import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeRegressor

from app.ml.pipeline import CAT_ATTRIBS, NUM_ATTRIBS, build_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "housing.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PIPELINE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

RANDOM_STATE = 42


def stratified_split(housing: pd.DataFrame):
    housing = housing.copy()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    for train_idx, test_idx in splitter.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_idx].drop(columns="income_cat").reset_index(drop=True)
        test_set = housing.loc[test_idx].drop(columns="income_cat").reset_index(drop=True)
    return train_set, test_set


def get_feature_names(pipeline):
    try:
        return list(pipeline.get_feature_names_out())
    except Exception:
        return NUM_ATTRIBS + CAT_ATTRIBS


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Loading dataset from {DATA_PATH} ...")
    housing = pd.read_csv(DATA_PATH)
    dataset_rows = len(housing)

    train_set, test_set = stratified_split(housing)

    y_train = train_set["median_house_value"].copy()
    X_train = train_set.drop(columns="median_house_value")
    y_test = test_set["median_house_value"].copy()
    X_test = test_set.drop(columns="median_house_value")

    pipeline = build_pipeline(NUM_ATTRIBS, CAT_ATTRIBS)
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    candidates = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=4,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    cv_results = {}
    print("Running 10-fold cross-validation for each candidate model ...")
    for name, estimator in candidates.items():
        scores = cross_val_score(
            estimator, X_train_prepared, y_train,
            scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1,
        )
        rmse_scores = -scores
        cv_results[name] = {
            "rmse_mean": float(rmse_scores.mean()),
            "rmse_std": float(rmse_scores.std()),
            "folds": [float(s) for s in rmse_scores],
        }
        print(f"  {name:18s} CV RMSE = {rmse_scores.mean():,.0f} (+/- {rmse_scores.std():,.0f})")

    best_name = min(cv_results, key=lambda k: cv_results[k]["rmse_mean"])
    print(f"Best candidate by CV RMSE: {best_name}")

    final_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    final_model.fit(X_train_prepared, y_train)
    train_seconds = time.time() - t0

    test_predictions = final_model.predict(X_test_prepared)
    test_rmse = float(root_mean_squared_error(y_test, test_predictions))
    test_mae = float(mean_absolute_error(y_test, test_predictions))
    test_r2 = float(r2_score(y_test, test_predictions))

    feature_names = get_feature_names(pipeline)
    importances = final_model.feature_importances_
    feature_importance = sorted(
        [{"feature": f, "importance": float(i)} for f, i in zip(feature_names, importances)],
        key=lambda d: d["importance"],
        reverse=True,
    )

    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(y_test), size=min(300, len(y_test)), replace=False
    )
    scatter_sample = {
        "actual": [float(v) for v in y_test.values[sample_idx]],
        "predicted": [float(v) for v in test_predictions[sample_idx]],
    }

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)

    metrics = {
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "dataset_rows": int(dataset_rows),
        "train_rows": int(len(train_set)),
        "test_rows": int(len(test_set)),
        "train_seconds": round(train_seconds, 2),
        "model_type": "RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=4)",
        "cv_results": cv_results,
        "best_cv_candidate": best_name,
        "test_metrics": {"rmse": test_rmse, "mae": test_mae, "r2": test_r2},
        "feature_importance": feature_importance,
        "scatter_sample": scatter_sample,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test RMSE={test_rmse:,.0f}  MAE={test_mae:,.0f}  R2={test_r2:.4f}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved pipeline -> {PIPELINE_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    train()
