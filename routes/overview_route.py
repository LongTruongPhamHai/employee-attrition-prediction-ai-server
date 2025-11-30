from fastapi import APIRouter
from services.overview_service import overview_service

router = APIRouter()


@router.get("/overview")
def overview_route():
    return overview_service()
