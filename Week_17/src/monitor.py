"""Inject known drift, evaluate it with Evidently, and log the report to MLflow."""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from data import DATA_PATH, load_raw_data

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    data = load_raw_data(DATA_PATH)
    reference = data.sample(frac=0.7, random_state=42).copy()
    current = data.drop(reference.index).copy()
    rng = np.random.default_rng(42)
    current["MonthlyCharges"] += 35.0 + rng.normal(0, 2, len(current))
    month_to_month = current[current["Contract"] == "Month-to-month"]
    current = pd.concat([current, month_to_month.sample(frac=1.5, replace=True, random_state=42)], ignore_index=True)
    current.loc[current.sample(frac=0.08, random_state=42).index, "Churn"] ^= 1
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    REPORTS.mkdir(exist_ok=True)
    report_path = REPORTS / "evidently_drift_report.html"
    snapshot.save_html(str(report_path))
    mean_charge_shift = float(current["MonthlyCharges"].mean() - reference["MonthlyCharges"].mean())
    reference_churn = float(reference["Churn"].mean())
    current_churn = float(current["Churn"].mean())
    churn_shift = current_churn - reference_churn
    custom_metrics = pd.DataFrame([{
        "metric": "monthly_charges_mean_shift", "value": mean_charge_shift,
        "reference_churn_rate": reference_churn, "current_churn_rate": current_churn,
        "target_churn_rate_shift": churn_shift,
    }])
    custom_path = REPORTS / "custom_drift_metrics.csv"
    custom_metrics.to_csv(custom_path, index=False)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    with mlflow.start_run(run_name="evidently-monitoring"):
        mlflow.log_artifact(report_path, artifact_path="evidently")
        mlflow.log_artifact(custom_path, artifact_path="evidently")
        mlflow.log_metrics({"monthly_charges_mean_shift": mean_charge_shift, "target_churn_rate_shift": churn_shift})
        mlflow.set_tag("interpretation", "Injected MonthlyCharges and Contract drift should be detected; retrain if production thresholds are exceeded.")
    print(custom_metrics.to_string(index=False))
    print(f"Saved Evidently report: {report_path}")


if __name__ == "__main__":
    main()
