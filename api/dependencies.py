from functools import lru_cache
from src.churn_ml.config import Settings, settings
from src.churn_ml.inference.predictor import ChurnPredictor, get_predictor


@lru_cache()
def get_settings() -> Settings:
    return settings


def get_model_predictor() -> ChurnPredictor:
    return get_predictor()
