import joblib


def save_model(pipeline, path):
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def load_model(path):
    return joblib.load(path)
