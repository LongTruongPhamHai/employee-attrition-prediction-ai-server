import os

# RAW_DATA = r"D:\DEV\.Projects\CodeDream\AI-ML-DL-Course\employee-attrition-prediction\ai-module\data\raw\WA_Fn-UseC_-HR-Employee-Attrition.csv"
# PROCESSED_DATA_DIR = r"D:\DEV\.Projects\CodeDream\AI-ML-DL-Course\employee-attrition-prediction\ai-module\data\processed"
# MODEL_DIR = r"D:\DEV\.Projects\CodeDream\AI-ML-DL-Course\employee-attrition-prediction\ai-module\models"
# BEST_MODEL_PATH = r"D:\DEV\.Projects\CodeDream\AI-ML-DL-Course\employee-attrition-prediction\ai-module\models\best_model.pkl"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(BASE_DIR, "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

RANDOM_STATE = 42
TEMP_SIZE = 0.4
TEST_SIZE = 0.5


RANDOM_STATE = 42
TEMP_SIZE = 0.4
TEST_SIZE = 0.5
