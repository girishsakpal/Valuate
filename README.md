# Valuate: California Housing Valuation

An end-to-end machine learning application: a scikit-learn regression pipeline
trained on the classic 1990 California Census housing dataset, served through
a Flask app with a real persistence layer (SQLite in development, PostgreSQL
in production), a prediction log for monitoring, and an interactive
data-exploration dashboard. Visual design is an architectural blueprint /
land-survey theme, deliberately not another purple-gradient AI dashboard.

## What it demonstrates

- **Modeling**: stratified train/test split, a `ColumnTransformer` pipeline
  (median imputation + scaling + one-hot encoding), 10-fold cross-validated
  comparison of three regressors, and a final Random Forest evaluated on a
  held-out test set (RMSE, MAE, R², feature importances).
- **Serving**: the fitted pipeline + model are persisted with `joblib` and
  loaded once per process; a clean `predict_one()` function is the single
  seam between the ML code and the web layer.
- **Data layer**: SQLAlchemy models with a config that switches from SQLite
  (zero-setup local dev) to PostgreSQL (`DATABASE_URL` env var) without any
  code changes, the same pattern used in real deployments.
- **Product surface**: four pages (Survey / Appraise / Log / Model) plus a
  small JSON API (`/api/predict`, `/api/metrics`, `/api/predictions`) so the
  model is usable outside the browser too.



Visit `http://localhost:5000`. The four pages:

- **Survey** (`/`): dataset overview, a geographic scatter of prices across
  California, price distribution, and a feature correlation heatmap.
- **Appraise** (`/predict`): a live form that runs the model and logs the
  result.
- **Log** (`/history`): every appraisal ever requested, paginated.
- **Model** (`/insights`): the model card: CV comparison, feature
  importance, predicted-vs-actual plot, and a plain-English pipeline summary.
