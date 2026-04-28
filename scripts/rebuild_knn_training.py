from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def load_training_frame(source_model) -> tuple[pd.DataFrame, pd.Series]:
    feature_names = list(source_model.feature_names_in_)
    X = pd.DataFrame(source_model._fit_X, columns=feature_names)
    y = pd.Series(source_model._y, name="TH")
    return X, y


def choose_cv_folds(y: pd.Series) -> int:
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        return max(2, min(5, len(y)))
    return max(2, min(5, int(class_counts.min())))


def main() -> None:
    source_path = resolve_source_model()
    source_model = joblib.load(source_path)

    X, y = load_training_frame(source_model)

    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    training_dump = X.copy()
    training_dump["TH"] = y.values
    training_dump.to_csv(OUTPUT_DATA, index=False)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    cv_folds = choose_cv_folds(y_train)
    if stratify is not None:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=max(2, min(5, len(X_train))), shuffle=True, random_state=42)

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="accuracy",
    )
    search.fit(X_train, y_train)

    rebuilt_model = search.best_estimator_
    rebuilt_model.fit(X_train, y_train)

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rebuilt_model, OUTPUT_MODEL)

    source_train_predictions = source_model.predict(X_train)
    source_test_predictions = source_model.predict(X_test)
    rebuilt_train_predictions = rebuilt_model.predict(X_train)
    rebuilt_test_predictions = rebuilt_model.predict(X_test)

    print(f"Source model: {source_path}")
    print(f"Rebuilt model: {OUTPUT_MODEL}")
    print(f"Extracted training data: {OUTPUT_DATA}")
    print(f"Train accuracy (source):  {accuracy_score(y_train, source_train_predictions):.6f}")
    print(f"Test accuracy (source):   {accuracy_score(y_test, source_test_predictions):.6f}")
    print(f"Train accuracy (rebuilt):  {accuracy_score(y_train, rebuilt_train_predictions):.6f}")
    print(f"Test accuracy (rebuilt):   {accuracy_score(y_test, rebuilt_test_predictions):.6f}")
    print(f"Best params: {search.best_params_}")
    print("Confusion matrix (test):\n", confusion_matrix(y_test, rebuilt_test_predictions))
    print("Classification report (test):\n", classification_report(y_test, rebuilt_test_predictions, digits=4, zero_division=0))


if __name__ == "__main__":
    main()