from fastapi import APIRouter
from services.train_service import train_services

router = APIRouter()


@router.get("/train-model")
def train_route():
    res = train_services()
    return res
