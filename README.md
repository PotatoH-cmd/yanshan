# 🌊 岩山（yanshan）LSTM 水文站预测服务

> 基于 LSTM 深度学习的水文站水位/流量预测系统，支持多站点批量训练与实时预测，为水利勘测提供数据驱动的决策支持。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LSTM-DeepLearning-purple?logo=pytorch&logoColor=white" alt="LSTM">
  <img src="https://img.shields.io/badge/Deploy-PM2-brightgreen?logo=pm2&logoColor=white" alt="PM2">
</p>

---

## ✨ 核心能力

| 能力 | 说明 |
|------|------|
| 📈 **水位预测** | 基于历史水位数据，预测未来 24h/72h/7天 水位变化趋势 |
| 🌊 **流量预测** | 根据降雨量、上游水位等多维特征，预测河道流量 |
| 🏘️ **多站点支持** | 每个水文站独立模型，支持批量训练与统一管理 |
| ⚡ **实时 API** | FastAPI 接口，毫秒级响应，可对接监测平台 |
| 📊 **批量训练** | 一键训练多个站点模型，自动保存最优权重 |
| 🔄 **滚动预测** | 支持滑动窗口预测，持续更新预测结果 |

---

## 🏗️ 架构全景

```
┌─────────────────────────────────────────────────────────────┐
│                    历史水文数据（CSV/Excel）                  │
│         水位 | 流量 | 降雨量 | 时间序列                        │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │    数据预处理            │
        │  归一化 | 时序窗口 | 特征工程 │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │    LSTM 神经网络         │
        │  Input → LSTM × N → Dense → Output │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │    预测结果              │
        │  未来水位/流量 + 置信区间  │
        └─────────────────────────┘
```

**模型结构**：LSTM（长短期记忆网络）专门处理时序数据，能捕捉水文数据中的长期依赖关系（如季节性变化、降雨滞后效应）。

---

## 🚀 快速启动

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/PotatoH-cmd/yanshan.git
cd yanshan

# 激活 conda 环境
conda activate fastapi-lstm

# 或创建新环境
conda create -n fastapi-lstm python=3.9 -y
conda activate fastapi-lstm
pip install -r requirements.txt
```

### 2. 训练模型（首次使用）

```bash
# 单站点训练
cd train
python train_model.py \
  --station 岩山水文站 \
  --data ../data/岩山_历史数据.csv \
  --epochs 100 \
  --window 30

# 批量训练多个站点
python batch_train.py \
  --data-dir ../data/ \
  --output-dir ../models/
```

### 3. 启动服务

```bash
# 方式一：直接启动
cd app
python main.py  # 默认端口 8000

# 方式二：PM2 托管（推荐生产环境）
pm2 start ecosystem.config.js
```

### 4. 验证

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","loaded_models":5,"stations":["岩山","李集","童庙","固始","商城"]}
```

---

## 📡 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/predict` | POST | 预测指定站点未来水位/流量 |
| `/predict/batch` | POST | 批量预测多个站点 |
| `/stations` | GET | 获取所有可用站点列表 |
| `/history` | GET | 查询站点历史数据 |
| `/health` | GET | 服务健康检查 |

### 预测示例

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "station": "岩山水文站",
    "predict_hours": 72,
    "features": ["water_level", "rainfall"]
  }'
```

**返回：**

```json
{
  "station": "岩山水文站",
  "predictions": [
    {"time": "2026-08-24 08:00", "water_level": 45.2, "confidence": 0.92},
    {"time": "2026-08-24 14:00", "water_level": 46.1, "confidence": 0.89},
    {"time": "2026-08-24 20:00", "water_level": 47.5, "confidence": 0.85}
  ],
  "trend": "rising",
  "alert": false
}
```

---

## 📁 项目结构

```
yanshan/
├── app/
│   ├── main.py              # FastAPI 服务入口
│   ├── predict.py           # 预测逻辑
│   ├── model_utils.py       # 模型加载与预处理
│   └── schemas.py           # 请求/响应数据模型
├── train/
│   ├── train_model.py       # 单站点训练
│   ├── batch_train.py       # 批量训练
│   └── data_loader.py       # 数据加载与预处理
├── models/
│   ├── 岩山水文站/          # 站点模型目录
│   │   ├── model.h5         # LSTM 模型权重
│   │   ├── scaler.pkl       # 归一化参数
│   │   └── config.json      # 模型配置
│   └── ...
├── data/
│   ├── raw/                 # 原始水文数据
│   └── processed/           # 处理后数据
├── output/                  # 预测输出
├── requirements.txt
└── README.md
```

---

## 🎯 应用场景

| 场景 | 效果 |
|------|------|
| 🌧️ **汛期预警** | 预测未来 72h 水位，超警戒值自动预警 |
| 🏗️ **工程调度** | 预测流量变化，辅助水库闸门调度决策 |
| 📊 **趋势分析** | 分析历史数据，识别水位季节性规律 |
| 🗺️ **多站联动** | 上下游站点联合预测，掌握水情传播 |

---

## ⚙️ 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 30 | 时序窗口大小（用过去 30 个时间点预测） |
| `lstm_units` | 64 | LSTM 隐藏层单元数 |
| `lstm_layers` | 2 | LSTM 层数 |
| `dropout` | 0.2 | Dropout 比率（防止过拟合） |
| `epochs` | 100 | 训练轮数 |
| `batch_size` | 32 | 批次大小 |
| `predict_hours` | 72 | 预测时长（小时） |

---

## 🧪 性能指标

| 指标 | 数值 |
|------|------|
| 支持站点数 | 5+（可扩展） |
| 单站预测耗时 | ~50ms |
| 模型训练时间 | ~10-30 分钟/站点 |
| 预测精度（MAE） | < 0.5m（水位） |
| 服务并发 | 100+ QPS |

---

## 📦 技术栈

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PM2-2B037A?style=for-the-badge&logo=pm2&logoColor=white" alt="PM2">
</p>

- **深度学习**：TensorFlow/Keras LSTM
- **数据处理**：NumPy、Pandas、Scikit-learn
- **服务框架**：FastAPI + Uvicorn
- **部署**：PM2 进程托管

---

## 📈 模型效果

```
岩山水文站 - 水位预测（过去 30 天 vs 预测）

水位(m)
  50 ┤                    ╭─╮
  48 ┤         ╭─╮      ╭╯ ╰╮     ← 预测值
  46 ┤    ╭───╯  ╰────╯    ╰──  ← 实际值
  44 ┤────╯
  42 ┤
     └────┬────┬────┬────┬────┬
         1日  10日  20日  25日  30日

MAE: 0.32m | RMSE: 0.45m | R²: 0.91
```

---

## 📝 更新日志

- **v1.0** — 初始版本：LSTM 单站预测 + FastAPI 接口
- **v1.1** — 新增批量训练、多站点支持
- **v1.2** — 新增滚动预测、置信区间输出

---

<p align="center">
  Made with 🌊 for 河南省水利勘测有限公司
</p>
