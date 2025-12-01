from common.utils import load_model_by_name
from config.settings import MODEL_DIR
import pandas as pd


def predict_service(new_data: dict, model_name: str):
    import pandas as pd
    from common.utils import load_model_by_name
    from config.settings import MODEL_DIR

    df = pd.DataFrame([new_data])
    model = load_model_by_name(MODEL_DIR, model_name)

    pred = model.predict(df)[0]

    if hasattr(pred, "item"):
        pred = pred.item()

    proba = None
    if hasattr(model, "predict_proba"):
        proba_array = model.predict_proba(df)[0]
        proba = proba_array.tolist()

    return {"model": model_name, "prediction": pred, "probability": proba}
