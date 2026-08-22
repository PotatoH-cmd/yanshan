from train_model import train_station

stations = ["station123", "station456"]
time_steps_dict = {"station123": 10, "station456": 12}

for sid in stations:
    try:
        train_station(station_id=sid, mode="vertical", time_steps=time_steps_dict.get(sid, 10))
    except Exception as e:
        print(f"❌ Error training {sid}: {e}")
