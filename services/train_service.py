import pandas as pd
from config.settings import (
    RAW_DATA,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEMP_SIZE,
    TEST_SIZE,
    MODEL_DIR,
    BEST_MODEL_PATH,
)
from common.overview import raw_data_overview
from sklearn.model_selection import train_test_split
from common.preprocess import data_preprocess
from common.train_models import train_models
from common.utils import save_model
import json
import numpy as np


def make_json_friendly(results):
    friendly = {}
    for model_name, res in results.items():
        friendly[model_name] = res.copy()
        if "pred_val" in res:
            friendly[model_name]["pred_val"] = res["pred_val"].tolist()
        if "pipeline" in res:
            friendly[model_name]["pipeline"] = str(res["pipeline"])
        if "val" in res:
            friendly[model_name]["val"] = {k: float(v) for k, v in res["val"].items()}
    return friendly


def train_services():
    df_raw = pd.read_csv(RAW_DATA)
    raw_data_overview(df_raw, "Raw data", "Attrition")

    X = df_raw.drop("Attrition", axis=1)
    y = df_raw["Attrition"].map({"Yes": 1, "No": 0})

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEMP_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )

    preprocess = data_preprocess(X_train)

    results = train_models(preprocess, X_train, y_train, X_val, y_val)

    best_model_name = max(results, key=lambda k: results[k]["val"]["F1"])
    best_pipeline = results[best_model_name]["pipeline"]

    print(
        f"Best model: {best_model_name}, "
        f"F1: {results[best_model_name]['val']['F1']:.4f}, "
    )

    save_model(best_pipeline, BEST_MODEL_PATH)

    results_json_ready = make_json_friendly(results)

    return {
        "best_model_name": best_model_name,
        "best_model_metrics": results_json_ready[best_model_name]["val"],
        "model_path": BEST_MODEL_PATH,
        "all_results": results_json_ready,
    }


if __name__ == "__main__":
    result = train_services()
    print(json.dumps(result, indent=4))
