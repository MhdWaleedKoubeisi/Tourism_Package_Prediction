# Tourism Package Prediction - MLOps pipeline

"Visit with Us" wants to know **which customers are likely to buy the new Wellness Tourism Package**.
This repository holds an automated MLOps pipeline (GitHub Actions) and a live prediction app (Streamlit Community Cloud).

## Repository structure

```
.
├── .github/workflows/pipeline.yml      # CI/CD: register data -> prepare data -> train + track -> commit model
├── tourism_project/
│   ├── data/tourism.csv                # registered raw dataset
│   ├── model_building/
│   │   ├── data_register.py            # validates and registers the dataset
│   │   ├── prep.py                     # cleaning + stratified train/test split
│   │   └── train.py                    # XGBoost grid search, MLflow tracking, saves best model
│   ├── deployment/
│   │   ├── app.py                      # Streamlit app
│   │   ├── requirements.txt            # dependencies for the Streamlit app
│   │   └── best_tourism_package_model_v1.joblib   # committed by the pipeline
│   └── requirements.txt                # dependencies for the pipeline
└── README.md
```

## How the pipeline works

1. **register-dataset** - checks the raw CSV (columns, unique IDs, binary target) and uploads it as an artifact.
2. **data-prep** - drops the index/ID columns, fixes the `Fe Male` typo, removes duplicate customers, makes an 80/20 stratified split.
3. **model-training** - starts a local MLflow server, tunes an XGBoost pipeline with 5-fold `GridSearchCV` (96 combinations, all logged),
   evaluates on the hold-out set, saves the best model and **commits it back to `main`**.
4. **Streamlit Community Cloud** serves `tourism_project/deployment/app.py` straight from this repo; every new model commit is picked up automatically.

Every push to `main` (or a manual *Run workflow*) re-runs the whole pipeline.

## Deploying the app

Streamlit Community Cloud -> *Create app* -> repository = this repo, branch = `main`,
main file = `tourism_project/deployment/app.py`, Python version = 3.11.
