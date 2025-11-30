from common.utils import load_model
from config.settings import MODEL_DIR
import pandas as pd


def predict_service(new_data, model_name):
    model_path = ""

    # match(model_name):
    #     case ''

    pipeline = load_model(MODEL_DIR)

    y_pred = pipeline.predict(new_data)

    return result
