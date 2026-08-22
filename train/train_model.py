import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from joblib import dump
from app.model_utils import n_features, feature_columns, get_model_paths
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
data_file = "data/data_with_qw5.xlsx"


def get_station_list_from_excel(data_file):
    df = pd.read_excel(data_file)
    stations = df['cezhan'].dropna().unique().tolist()  # 去重并转成列表
    return stations

def batch_train(station_list, mode, time_steps):
    for station_id in station_list:
        try:
            print(f"⏳ Training station {station_id} ...")
            train_station(station_id, mode, time_steps)
        except Exception as e:
            print(f"⚠️ Failed for {station_id}: {e}")


def build_model(time_steps):
    model = Sequential([
        LSTM(128, input_shape=(time_steps, n_features)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

def train_station(station_id, mode, time_steps):
    label_column = 'y0' if mode == 'vertical' else 'y1'
    df = pd.read_excel(data_file)
    df = df[df['cezhan'] == station_id]

    df = df.dropna(subset=feature_columns + [label_column])
    if df.shape[0] < time_steps + 1:
        raise ValueError(f"{station_id}: Not enough data.")

    data_X = df[feature_columns].values
    data_Y = df[label_column].values

    X_seq, y_seq = [], []
    for i in range(len(data_X) - time_steps):
        X_seq.append(data_X[i:i + time_steps])
        y_seq.append(data_Y[i + time_steps - 1])

    X = np.array(X_seq)
    y = np.array(y_seq).reshape(-1, 1)

    model = build_model(time_steps)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y)

    callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

    model.fit(X_scaled, y_scaled, epochs=128, batch_size=4, verbose=1)

    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_scaled)
    r2 = r2_score(y_true, y_pred)

    model_path, scaler_X_path, scaler_y_path = get_model_paths(station_id, mode)
    model.save(model_path)
    dump(scaler_X, scaler_X_path)
    dump(scaler_y, scaler_y_path)

    print(f"✅ Training complete for {station_id} | R²: {r2:.4f}")

def batch_train(station_list, mode, time_steps):
    for station_id in station_list:
        try:
            print(f"⏳ Training station {station_id} ...")
            train_station(station_id, mode, time_steps)
        except Exception as e:
            print(f"⚠️ Failed for {station_id}: {e}")

def train_single_station(station_id: str, mode: str = "vertical", time_steps: int = 6):
    """
    Train model for a single station.

    Args:
        station_id (str): The station identifier (e.g. '站点A')
        mode (str): 'vertical' or 'horizontal', decides label column
        time_steps (int): Number of time steps in input sequence
    """
    print(f"📌 Starting single station training for: {station_id}")
    train_station(station_id, mode, time_steps)

if __name__ == "__main__":
    #data_file = "data/data_with_qw5.xlsx"
    #stations = get_station_list_from_excel(data_file)
    #print(f"Found {len(stations)} unique stations.")
    #batch_train(stations, mode="vertical", time_steps=6)
    train_single_station("LD15-1", mode="horizontal", time_steps=6)