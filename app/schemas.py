from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    station_id: str
    mode: str  # "train" or "test"
    X: List[List[List[float]]]  # 3D list
