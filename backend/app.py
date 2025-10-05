# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")  # optional

# --- Load model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Optionally load LabelEncoder (if used)
label_encoder = None
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

# --- Initialize app ---
app = FastAPI(title="Exoplanet Classifier API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel, Field

class PlanetInput(BaseModel):
    koi_pdisposition: str = Field(..., example="CANDIDATE")
    koi_score: float = Field(..., example=0.95)
    koi_fpflag_nt: int = Field(..., example=0)
    koi_fpflag_ss: int = Field(..., example=0)
    koi_fpflag_co: int = Field(..., example=0)
    koi_fpflag_ec: int = Field(..., example=0)
    koi_period: float = Field(..., example=10.5)
    koi_time0bk: float = Field(..., example=134.56)
    koi_impact: float = Field(..., example=0.5)
    koi_duration: float = Field(..., example=2.5)
    koi_depth: float = Field(..., example=350.0)
    koi_prad: float = Field(..., example=1.2)
    koi_teq: float = Field(..., example=550.0)
    koi_insol: float = Field(..., example=1.5)
    koi_model_snr: float = Field(..., example=12.3)
    koi_tce_plnt_num: int = Field(..., example=1)
    koi_tce_delivname: str = Field(..., example="q1_q17_dr25")
    koi_steff: float = Field(..., example=5700.0)
    koi_slogg: float = Field(..., example=4.4)
    koi_srad: float = Field(..., example=1.0)
    ra: float = Field(..., example=291.9)
    dec: float = Field(..., example=48.14)
    koi_kepmag: float = Field(..., example=15.3)

# Map numeric labels to human-readable outputs
LABEL_MAP = {
    0: "FALSE POSITIVE",
    1: "CONFIRMED",
    2: "CANDIDATE"
}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(inp: PlanetInput):
    try:
        # Column order must match training dataset
        cols = [
            'koi_pdisposition', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss',
            'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period', 'koi_time0bk',
            'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
            'koi_insol', 'koi_model_snr', 'koi_tce_plnt_num', 'koi_tce_delivname',
            'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag'
        ]

        # Convert input into a single-row DataFrame
        X = pd.DataFrame([[getattr(inp, c) for c in cols]], columns=cols)

        # Convert string columns to numeric if label encoded during training
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]

        # Run prediction
        pred = model.predict(X)[0]

        # Prediction confidence (if supported)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(np.max(model.predict_proba(X)[0]))

        label = LABEL_MAP.get(int(pred), str(pred))

        return {
            "prediction": label,
            "prediction_code": int(pred),
            "confidence": proba
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Add to bottom of app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)