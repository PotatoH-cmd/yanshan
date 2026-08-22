import pandas as pd
import numpy as np
from app.model_utils import load_station_model, n_features, feature_columns
from datetime import datetime

# -------- 配置 --------
station_id = "LD12-4"
mode = "horizontal"  # 或 horizontal
label_column = "y0" if mode == "vertical" else "y1"
time_steps = 6
data_file = "data/data_with_qw5.xlsx"

# -------- 手动输入未来一天特征 --------
manual_input = {
    't': 4749.0,
    'bjgc': 88.1,
    'mean_RZ': 106,
    'mean_W': 85,
    'temperature': 25.6
}
manual_time = "2024-03-01 00:00:00"

# -------- 时间点（前5天从Excel取，第6天为手动输入） --------
target_dates = [
    "2022-12-01 00:00:00",
    "2023-03-01 00:00:00",
    "2023-06-01 00:00:00",
    "2023-09-01 00:00:00",
    "2023-12-01 00:00:00",
]

# -------- 读取数据 --------
df = pd.read_excel(data_file)
df = df[df['cezhan'] == station_id]
df['date'] = pd.to_datetime(df['month'])

# -------- 提取前5条时间点特征 --------
selected_rows = []
observed_y = []
for d in target_dates:
    row = df[df['date'] == pd.to_datetime(d)]
    if row.empty:
        raise ValueError(f"缺少时间点 {d} 的数据")
    selected_rows.append(row[feature_columns].iloc[0].tolist())
    observed_y.append((d, row[label_column].iloc[0]))


# -------- 添加人工输入第6天 --------
manual_row = [manual_input[col] for col in feature_columns]
selected_rows.append(manual_row)

# -------- 构造预测输入 --------
X_input = np.array(selected_rows).reshape(1, time_steps, n_features)

# -------- 加载模型并预测 --------
model, scaler_X, scaler_y = load_station_model(station_id, mode)
if model is None:
    raise FileNotFoundError(f"未找到模型，请先训练 {station_id} (mode: {mode})")

X_scaled = scaler_X.transform(X_input.reshape(-1, n_features)).reshape(X_input.shape)
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# -------- 打印结果 --------
print("\n观测值（变形量）:")
for date_str, obs_val in observed_y:
    print(f"{date_str}: {obs_val:.4f}")

print("\n预测值（变形量）:")
print(f"\n预测结果：{manual_time} @ {station_id} 预测变形量（{label_column}） = {y_pred.flatten()[0]:.4f}")

import matplotlib.pyplot as plt
import os

# -------- 可视化 --------
dates = [d for d, _ in observed_y] + [manual_time]
values = [v for _, v in observed_y] + [y_pred]

# 创建输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
image_path = os.path.join(output_dir, f"prediction_{station_id}_{label_column}_{mode}.png")

plt.figure(figsize=(10, 5))
plt.plot(dates[:5], values[:5], marker='o', label="Observed", color='blue')
plt.plot(dates[5], values[5], marker='x', markersize=10, label="Predicted", color='red')

for i in range(5):
    plt.text(dates[i], values[i], f"{values[i]:.2f}", ha='center', va='bottom', fontsize=9)
plt.text(dates[5], float(values[5]), f"{float(values[5]):.2f}", ha='center', va='bottom', fontsize=10, color='red')

plt.title(f"Station {station_id} Prediction ({label_column})")
plt.xlabel("Date")
plt.ylabel("Deformation")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True)

# 保存图像
plt.savefig(image_path)
plt.close()

print(f"\n预测图已保存至: {image_path}")
