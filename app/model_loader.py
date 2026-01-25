import joblib
from app.config import MODEL_PATH

def load_model():
    with open(MODEL_PATH, 'rb') as model_file:
        model = joblib.load(model_file)
    return model

model = load_model()