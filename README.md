# yanshan（岩山）LSTM 水文站预测服务

LSTM 水文站预测 API 服务（FastAPI），支持多站点水位/流量预测。

## 项目结构

- `app/` — FastAPI 服务（main.py 入口，predict.py 预测逻辑，model_utils.py 模型加载）
- `train/` — LSTM 模型训练脚本（train_model.py、batch_train.py）
- `models/` — 训练好的模型权重（.h5）与归一化参数（.pkl）
- `data/` — 训练/预测数据
- `output/` — 预测输出

## 启动

```bash
# 需 conda 环境 fastapi-lstm
conda activate fastapi-lstm
cd app && python main.py   # 默认端口 8000
```
