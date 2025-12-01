import joblib
import os


def save_model(pipeline, path):
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def load_model(path):
    return joblib.load(path)


def load_model_by_name(model_dir, model_name):
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found in {model_dir}")

    return joblib.load(model_path)
