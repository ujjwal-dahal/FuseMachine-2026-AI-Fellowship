"""Train three churn models, log evidence to MLflow, and register the winner."""
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data import DATA_PATH, load_raw_data, make_preprocessor, split_features_target

ROOT = Path(__file__).resolve().parents[1]
MLRUNS = ROOT / "mlruns"
REPORTS = ROOT / "reports"
MODEL_NAME = "TelcoChurnBest"


def build_models() -> list[tuple[str, object, dict]]:
    return [
        ("logistic_c1", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="liblinear"), {"C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 1000}),
        ("logistic_c05", LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", solver="liblinear"), {"C": 0.5, "solver": "liblinear", "class_weight": "balanced", "max_iter": 1000}),
        ("random_forest", RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1), {"n_estimators": 250, "max_depth": 8, "min_samples_leaf": 3, "class_weight": "balanced", "random_state": 42}),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42), {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}),
    ]


def log_plots(name: str, y_true: pd.Series, predictions, probabilities) -> None:
    REPORTS.mkdir(exist_ok=True)
    matrix = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix: {name}")
    plt.tight_layout(); plt.savefig(REPORTS / f"confusion_matrix_{name}.png", dpi=140); plt.close()
    false_positive, true_positive, _ = roc_curve(y_true, probabilities)
    plt.figure(figsize=(5, 4)); plt.plot(false_positive, true_positive, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False positive rate")
    plt.ylabel("True positive rate"); plt.title(f"ROC curve: {name}"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / f"roc_curve_{name}.png", dpi=140); plt.close()


def main() -> None:
    mlflow.set_tracking_uri(MLRUNS.as_uri())
    experiment = mlflow.set_experiment("telco-churn-experiments")
    data = load_raw_data(DATA_PATH)
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
    rows = []
    best = None
    for name, estimator, parameters in build_models():
        pipeline = Pipeline([("preprocessor", make_preprocessor(x_train)), ("model", estimator)])
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=name) as run:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            probabilities = pipeline.predict_proba(x_test)[:, 1]
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
            mlflow.log_params(parameters); mlflow.log_metrics(metrics)
            log_plots(name, y_test, predictions, probabilities)
            mlflow.log_artifact(REPORTS / f"confusion_matrix_{name}.png", artifact_path="plots")
            mlflow.log_artifact(REPORTS / f"roc_curve_{name}.png", artifact_path="plots")
            signature = infer_signature(x_train, pipeline.predict(x_train))
            mlflow.sklearn.log_model(pipeline, "model", signature=signature, input_example=x_train.head(2), registered_model_name=MODEL_NAME)
            rows.append({"run_name": name, "run_id": run.info.run_id, **metrics, **parameters})
            candidate = (metrics["f1"], metrics["roc_auc"], pipeline, run.info.run_id, metrics)
            if best is None or candidate[:2] > best[:2]: best = candidate
    comparison = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False)
    comparison.to_csv(REPORTS / "run_comparison.csv", index=False)
    print(comparison[["run_name", "run_id", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    client = mlflow.tracking.MlflowClient(tracking_uri=MLRUNS.as_uri())
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    winner_version = max(versions, key=lambda version: int(version.version))
    client.transition_model_version_stage(MODEL_NAME, winner_version.version, "Staging", archive_existing_versions=True)
    client.transition_model_version_stage(MODEL_NAME, winner_version.version, "Production", archive_existing_versions=True)
    print(f"Registered {MODEL_NAME} version {winner_version.version} from run {best[3]}; Staging -> Production")


if __name__ == "__main__":
    main()
