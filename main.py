from fastapi import FastAPI
from routes.train_route import router as train_route
from routes.overview_route import router as overview_route

app = FastAPI()

app.include_router(overview_route, prefix="/api/overview-data", tags=["Overview data"])
app.include_router(train_route, prefix="/api/train-model", tags=["Train model"])
