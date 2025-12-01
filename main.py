from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.train_route import router as train_route
from routes.overview_route import router as overview_route
from routes.predict_route import router as predict_route

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(overview_route, prefix="/api/overview-data", tags=["Overview data"])
app.include_router(train_route, prefix="/api/train-model", tags=["Train model"])
app.include_router(predict_route, prefix="/api/predict", tags=["Predict"])
