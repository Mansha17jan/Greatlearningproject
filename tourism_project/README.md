# Tourism MLOps Project

Predicts whether a customer will purchase the Wellness Tourism Package.

## Pipeline
1. Register tourism.csv on Hugging Face Hub.
2. Prepare/clean data and create stratified train/test sets.
3. Train a Random Forest classifier and track parameters/metrics with MLflow.
4. Build a Streamlit application.
5. GitHub Actions automatically deploys the application to a Hugging Face Streamlit Space.

## Expected repository structure
- data/tourism.csv
- model_building/data_prep.py
- model_building/train.py
- deployment/app.py
- deployment/model.joblib
- deployment/requirements.txt
- deployment/Dockerfile
- .github/workflows/pipeline.yml
