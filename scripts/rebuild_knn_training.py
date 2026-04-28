from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "KNN_best_model_clean.pkl",
    PROJECT_ROOT / "models" / "KNN_best_model.pkl",
]
OUTPUT_MODEL = PROJECT_ROOT / "models" / "KNN_best_model_rebuilt.pkl"
OUTPUT_DATA = PROJECT_ROOT / "data" / "extracted" / "knn_training_data.csv"


def resolve_source_model() -> Path:
    for candidate in SOURCE_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a saved KNN model. Looked in: "
        + ", ".join(str(path) for path in SOURCE_MODEL_CANDIDATES)
    )


def main() -> None:
    source_path = resolve_source_model()
    source_model = joblib.load(source_path)

    feature_names = list(source_model.feature_names_in_)
    X = pd.DataFrame(source_model._fit_X, columns=feature_names)
    y = pd.Series(source_model._y, name="TH")

    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    training_dump = X.copy()
    training_dump["TH"] = y.values
    training_dump.to_csv(OUTPUT_DATA, index=False)

    rebuilt_model = KNeighborsClassifier(**source_model.get_params())
    rebuilt_model.fit(X, y)

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rebuilt_model, OUTPUT_MODEL)

    original_predictions = source_model.predict(X)
    rebuilt_predictions = rebuilt_model.predict(X)

    print(f"Source model: {source_path}")
    print(f"Rebuilt model: {OUTPUT_MODEL}")
    print(f"Extracted training data: {OUTPUT_DATA}")
    print(f"Training accuracy (source):  {accuracy_score(y, original_predictions):.6f}")
    print(f"Training accuracy (rebuilt): {accuracy_score(y, rebuilt_predictions):.6f}")
    print(f"Predictions identical: {bool((original_predictions == rebuilt_predictions).all())}")
    print("Confusion matrix:\n", confusion_matrix(y, rebuilt_predictions))
    print("Classification report:\n", classification_report(y, rebuilt_predictions, digits=4))


if __name__ == "__main__":
    main()