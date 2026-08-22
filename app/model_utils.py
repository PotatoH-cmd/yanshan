import os
from tensorflow.keras.models import load_model
from joblib import load

# ✅ 这两行是关键变量，必须添加
n_features = 5
feature_columns = ['t', 'bjgc', 'mean_RZ', 'mean_W', 'temperature']

model_directory = "models"

def get_model_paths(station_id, mode):
    suffix = "y0" if mode == "vertical" else "y1"
    return (
        os.path.join(model_directory, f"model_{station_id}_{suffix}.h5"),
        os.path.join(model_directory, f"scaler_X_{station_id}_{suffix}.pkl"),
        os.path.join(model_directory, f"scaler_y_{station_id}_{suffix}.pkl")
    )

def load_station_model(station_id, mode):
    model_path, scaler_X_path, scaler_y_path = get_model_paths(station_id, mode)
    if not os.path.exists(model_path):
        return None, None, None
    return (
        load_model(model_path),
        load(scaler_X_path),
        load(scaler_y_path)
    )
