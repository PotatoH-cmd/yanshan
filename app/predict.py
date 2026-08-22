from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import numpy as np
import json
from app.model_utils import n_features, load_station_model
from app.schemas import PredictRequest

router = APIRouter()

@router.post("/predict")
def predict_post(req: PredictRequest):
    return _run_prediction(req.X, req.station_id, req.mode)

@router.get("/predict")
def predict_get(
    X: str = Query(..., description="JSON-encoded 3D array"),
    station_id: str = Query(...),
    mode: str = Query(...)
):
    try:
        X_parsed = json.loads(X)
        return _run_prediction(X_parsed, station_id, mode)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON for X: {str(e)}")

def _run_prediction(X_input, station_id, mode):
    try:
        X = np.array(X_input)

        if X.shape[2] != n_features:
            raise ValueError(f"Each input must have {n_features} features per time step.")

        model, scaler_X, scaler_y = load_station_model(station_id, mode)
        if model is None:
            raise ValueError(f"Model not found for station '{station_id}' in mode '{mode}'.")

        X_scaled = scaler_X.transform(X.reshape(-1, n_features)).reshape(X.shape)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        return {"prediction": y_pred.flatten().tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
