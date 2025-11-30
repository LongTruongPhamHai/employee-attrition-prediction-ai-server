from fastapi import FastAPI
from routes.train_route import router as train_route

app = FastAPI()

app.include_router(train_route, prefix="/api/train-model", tags=["Train model"])
