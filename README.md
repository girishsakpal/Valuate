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
- **Ops**: `Dockerfile` + `docker-compose.yml` running the app behind
  Gunicorn against a real Postgres container.

## Project layout

```
housing-ml-app/
├── app/
│   ├── __init__.py          # app factory
│   ├── config.py            # Dev (SQLite) / Prod (PostgreSQL) config
│   ├── extensions.py        # db = SQLAlchemy()
│   ├── models.py            # Prediction model (the appraisal log)
│   ├── ml/
│   │   ├── pipeline.py      # shared preprocessing ColumnTransformer
│   │   ├── train.py         # training script -> model.pkl, pipeline.pkl, metrics.json
│   │   └── predictor.py     # loads artifacts, predict_one()
│   ├── routes/
│   │   ├── main.py          # /EDA dashboard
│   │   ├── predict.py       # /predict, appraisal form
│   │   ├── history.py       # /history, prediction log
│   │   ├── insights.py      # /insights, model card / metrics
│   │   └── api.py           # /api/* JSON endpoints
│   ├── static/{css,js}
│   └── templates/
├── data/housing.csv
├── model_artifacts/         # model.pkl, pipeline.pkl, metrics.json (generated)
├── run.py                   # local dev entrypoint
├── wsgi.py                  # gunicorn entrypoint
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quickstart: development (SQLite)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the model (writes model_artifacts/*.pkl + metrics.json)
python -m app.ml.train

# Run the app, SQLite db is created automatically at instance/housing_dev.db
python run.py
```

## Resume framing

A few ways to describe this project depending on the role you're targeting:

- *"Built and deployed an end-to-end ML system (Flask + scikit-learn +
  SQLAlchemy) that trains a Random Forest regressor on the California
  Housing dataset, serves live predictions through a REST API and web UI,
  and logs every inference to a database for monitoring, configured to run
  on SQLite locally and PostgreSQL in production via Docker Compose."*
- *"Compared three regression models with 10-fold cross-validation,
  selected and tuned a Random Forest (R² ≈ 0.83 on held-out data), and
  shipped it behind a Flask API with an interactive EDA dashboard
  (Plotly.js) for geographic and correlation analysis."*

Feel free to add real screenshots of the Survey and Model pages to your
portfolio/resume, they're the most visually distinctive parts of the app.
