import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_train = pd.read_csv("model_building/X_train.csv")
X_test = pd.read_csv("model_building/X_test.csv")
y_train = pd.read_csv("model_building/y_train.csv").squeeze("columns")
y_test = pd.read_csv("model_building/y_test.csv").squeeze("columns")

categorical = X_train.select_dtypes(include="object").columns.tolist()
numeric = X_train.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Tourism_Package_Purchase")

with mlflow.start_run(run_name="RandomForest"):
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    }
    mlflow.log_params({
        "model": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42
    })
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipeline, "model")

    os.makedirs("deployment", exist_ok=True)
    joblib.dump(pipeline, "deployment/model.joblib")

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
