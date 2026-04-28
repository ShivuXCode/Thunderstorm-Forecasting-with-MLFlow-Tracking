from collections.abc import Mapping, Sequence

import pandas as pd

from app.config import resolve_model_path
from app.model_loader import load_model


FEATURE_COLUMNS = [
    "SWEAT index",
    "K index",
    "Totals totals index",
    "Environmental_Stability",
    "Moisture_Indices",
    "Convective_Potential",
    "Temperature_Pressure",
    "Moisture_Temperature_Profiles",
]


def _build_input_frame(features: Sequence[float] | Mapping[str, float] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return features.loc[:, FEATURE_COLUMNS].copy()

    if isinstance(features, Mapping):
        missing = [column for column in FEATURE_COLUMNS if column not in features]
        if missing:
            raise ValueError(f"Missing required feature values: {missing}")
        return pd.DataFrame([[features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, received {len(features)}")

    return pd.DataFrame([list(features)], columns=FEATURE_COLUMNS)


def predict_weather(features: Sequence[float] | Mapping[str, float] | pd.DataFrame):
    model = load_model()
    frame = _build_input_frame(features)

    prediction = model.predict(frame)
    proba = model.predict_proba(frame)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "prediction": int(prediction[0]),
        "probability": float(proba[0]) if proba is not None else None,
        "model_path": str(resolve_model_path()),
    }