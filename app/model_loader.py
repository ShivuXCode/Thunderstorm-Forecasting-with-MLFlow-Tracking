from functools import lru_cache

import joblib

from app.config import resolve_model_path


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(resolve_model_path())


model = load_model()