from fastapi import APIRouter, HTTPException
from services.predict_service import predict_service

router = APIRouter()


@router.post("/predict")
def predict_endpoint(payload: dict):
    try:
        model_name = payload.get("model_name")
        data = payload.get("data")

        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")

        if not data:
            raise HTTPException(status_code=400, detail="data is required")

        result = predict_service(data, model_name)

        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
