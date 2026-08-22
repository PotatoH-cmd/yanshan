from fastapi import FastAPI
from app.predict import router as predict_router
import uvicorn
app = FastAPI(title="LSTM Station Prediction API")

# 注册路由
app.include_router(predict_router)

# 启动服务
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
#nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > output.log 2>&1 &