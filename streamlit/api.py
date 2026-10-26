"""API FastAPI de prédiction du rendement viticole.

Expose :
  - GET /            : ping
  - GET /historique  : données historiques 2010-2024 (table training_vigne)
  - GET /projections : projections CORDEX 2054/2084/2100 (table gold_projection_rendement_vigne)
  - POST /predict    : prédiction en direct avec le modèle XGBoost

À lancer depuis la racine du repo :
    python streamlit/api.py
"""

from pathlib import Path

import joblib
import os
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine

# --- Config : .env à la racine, .joblib à la racine ---
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

DATABASE_URL = os.getenv("NEON_DB_URL") or os.getenv("NEON_DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("NEON_DB_URL ou NEON_DATABASE_URL manquant dans .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# --- Chargement des artefacts ML (dans models) ---
MODELS_DIR = ROOT / "models"
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
xgb_model = joblib.load(MODELS_DIR / "xgb_model.joblib")
features = joblib.load(MODELS_DIR / "features.joblib")

# --- App ---
app = FastAPI(title="API Prédiction Rendement Vigne", version="2.0")


class PredictionInput(BaseModel):
    temp_saison: float
    precip_cumul: float
    solar_moy: float
    soil_family_dominant: str
    soil_type_dominant: str
    year: int
    code_departement: str


@app.get("/")
def home():
    return {"message": "API Rendement Vigne v2 — en ligne"}


@app.get("/historique")
def get_historique():
    """Historique 2010-2024 par département (source : training_vigne)."""
    query = """
        SELECT code_departement, year, rendement_q_ha,
               temp_saison, precip_cumul, solar_moy,
               soil_family_dominant, soil_type_dominant
        FROM training_vigne
        ORDER BY code_departement, year;
    """
    df = pd.read_sql(query, engine)
    df = df.dropna(subset=["rendement_q_ha"])
    return df.to_dict(orient="records")


@app.get("/projections")
def get_projections():
    """Projections CORDEX 2054/2084/2100 par département."""
    query = """
        SELECT code_departement, year,
               rendement_bas, rendement_moyen, rendement_haut,
               temp_saison, precip_cumul, solar_moy,
               soil_family_dominant, soil_type_dominant
        FROM gold_projection_rendement_vigne
        ORDER BY year, code_departement;
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")


@app.post("/predict")
def predict_rendement(data: PredictionInput):
    """Prédiction en direct via XGBoost."""
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[features]
    input_encoded = preprocessor.transform(input_df)
    prediction = float(xgb_model.predict(input_encoded)[0])
    prediction = max(0, prediction)
    return {
        "code_departement": data.code_departement,
        "year": data.year,
        "rendement_predit_q_ha": round(prediction, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)