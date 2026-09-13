from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import time
from pathlib import Path


# -----------------------------------
# Create FastAPI application
# -----------------------------------

app = FastAPI(
    title="AI-FraudGuard",
    description="Real-Time Credit Card Fraud Detection API using XGBoost",
    version="1.0.0"
)


# -----------------------------------
# Project paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
TEMPLATES_DIR = BASE_DIR / "templates"


# -----------------------------------
# HTML Templates
# -----------------------------------

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# -----------------------------------
# Load trained model
# -----------------------------------

model = joblib.load(MODEL_PATH)


# -----------------------------------
# Input data structure
# -----------------------------------

class Transaction(BaseModel):

    amount: float = Field(
        ...,
        gt=0
    )

    transaction_hour: int = Field(
        ...,
        ge=0,
        le=23
    )

    merchant_category: str

    foreign_transaction: int = Field(
        ...,
        ge=0,
        le=1
    )

    location_mismatch: int = Field(
        ...,
        ge=0,
        le=1
    )

    device_trust_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    velocity_last_24h: int = Field(
        ...,
        ge=0
    )

    cardholder_age: int = Field(
        ...,
        ge=18,
        le=100
    )


# -----------------------------------
# Main Web Interface
# -----------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# -----------------------------------
# Web Application
# -----------------------------------

@app.get("/web", response_class=HTMLResponse)
def web_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# -----------------------------------
# Health Check
# -----------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# -----------------------------------
# Fraud Prediction
# -----------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    start_time = time.perf_counter()

    # Create dataframe from user input
    data = pd.DataFrame([
        {
            "amount": transaction.amount,

            "transaction_hour":
                transaction.transaction_hour,

            "merchant_category":
                transaction.merchant_category,

            "foreign_transaction":
                transaction.foreign_transaction,

            "location_mismatch":
                transaction.location_mismatch,

            "device_trust_score":
                transaction.device_trust_score,

            "velocity_last_24h":
                transaction.velocity_last_24h,

            "cardholder_age":
                transaction.cardholder_age
        }
    ])

    # Make prediction
    prediction = model.predict(data)[0]

    # Get fraud probability
    probability = model.predict_proba(data)[0][1]

    # Calculate inference time
    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    # Convert prediction to readable result
    if prediction == 1:
        result = "Fraudulent"
    else:
        result = "Genuine"

    # Return API response
    return {
        "prediction": result,

        "fraud_probability":
            round(float(probability) * 100, 2),

        "inference_time_ms":
            round(inference_time, 3)
    }
```
