"""FastAPI service loading the MLflow Production-stage model."""
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
MODEL_URI = "models:/TelcoChurnBest/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)
app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict")
def predict(customer: Customer) -> dict[str, object]:
    row = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict(row)[0])
    return {"churn_prediction": "Yes" if probability >= 0.5 else "No", "churn_probability": probability}
