# Thunderstorm Predictor: Code Walkthrough and Project Flow

This document explains:
- what each file does,
- what content it contains,
- and how data flows through the project during prediction and training-rebuild.

## 1) Project Purpose

This project predicts thunderstorm occurrence (`0` or `1`) using a saved KNN classifier and a Streamlit UI.

There are two main workflows:
- **Inference workflow (UI prediction):** user enters weather features in Streamlit, app returns prediction and probability.
- **Training-rebuild workflow:** script reconstructs training data from the saved model internals and rebuilds an equivalent model.

## 2) Current Folder Layout

- `app.py`
- `requirements.txt`
- `PROJECT_FLOW.md`
- `app/`
  - `__init__.py`
  - `config.py`
  - `model_loader.py`
  - `predictor.py`
  - `sample_inputs.py`
- `scripts/`
  - `rebuild_knn_training.py`
- `models/`
  - `KNN_best_model_clean.pkl`

## 3) File-by-File Explanation

### `app.py`

**Purpose**
- Main Streamlit UI entrypoint.

**What it contains**
- Page configuration (`st.set_page_config`).
- Custom CSS styles for hero card, sidebar, and input/result cards.
- Sidebar preset selector and "Load preset" action.
- Input form built from `FEATURE_COLUMNS`.
- Calls `predict_weather(inputs)` when user submits.
- Displays:
  - thunderstorm label (`Thunderstorm likely` / `No thunderstorm likely`),
  - probability,
  - model path used.

**Why it matters**
- This is the visible app users interact with.

---

### `app/config.py`

**Purpose**
- Central model path configuration.

**What it contains**
- `PROJECT_ROOT` detection.
- `MODEL_PATH = PROJECT_ROOT / "models" / "KNN_best_model_clean.pkl"`.
- `resolve_model_path()` that validates existence and raises a clear error if missing.

**Why it matters**
- Keeps model-path logic in one place.

---

### `app/model_loader.py`

**Purpose**
- Loads and caches the model.

**What it contains**
- `load_model()` decorated with `@lru_cache(maxsize=1)`.
- Uses `joblib.load(resolve_model_path())`.
- Module-level `model = load_model()`.

**Why it matters**
- Prevents reloading the model repeatedly during app usage.

---

### `app/predictor.py`

**Purpose**
- Core prediction logic shared by UI and any future callers.

**What it contains**
- `FEATURE_COLUMNS` (the exact 8 required feature names).
- `_build_input_frame(...)`:
  - accepts `Sequence`, `Mapping`, or `pandas.DataFrame`,
  - validates required feature count/names,
  - converts input to a properly ordered single-row DataFrame.
- `predict_weather(...)`:
  - loads model,
  - predicts class (`predict`),
  - predicts probability (`predict_proba` when available),
  - returns dict with `prediction`, `probability`, and `model_path`.

**Why it matters**
- Guarantees consistent input format and correct feature order before inference.

---

### `app/sample_inputs.py`

**Purpose**
- Stores user-friendly preset input values used by the UI.

**What it contains**
- Same `FEATURE_COLUMNS` list used by UI/predictor.
- `SAMPLE_PRESETS` dictionary:
  - `Calm baseline`
  - `Storm-prone sample`
  - `High-risk sample`
- `DEFAULT_PRESET` selection.

**Why it matters**
- Makes the UI easier to use and quickly test realistic scenarios.

---

### `scripts/rebuild_knn_training.py`

**Purpose**
- Training-equivalent recovery script from the saved model internals.

**What it contains**
- Loads saved KNN model from `models/KNN_best_model_clean.pkl`.
- Extracts fitted training matrix and labels (`_fit_X`, `_y`).
- Recreates a new KNN with original hyperparameters (`get_params()`).
- Fits rebuilt model on extracted data.
- Saves outputs:
  - rebuilt model: `models/KNN_best_model_rebuilt.pkl`
  - extracted data: `data/extracted/knn_training_data.csv`
- Prints metrics and checks prediction identity with source model.

**Why it matters**
- Gives you a runnable `.py` training-equivalent path even though original training source was unavailable.

---

### `models/KNN_best_model_clean.pkl`

**Purpose**
- Primary persisted model used for prediction.

**What it contains**
- Serialized `KNeighborsClassifier` object.
- Includes trained state (neighbors data, labels, parameters, feature names).

**Why it matters**
- Required for both inference and rebuild script.

---

### `requirements.txt`

**Purpose**
- Runtime dependencies.

**What it contains**
- `streamlit`, `joblib`, `scikit-learn`, `numpy`, `pandas` (pinned versions).

**Why it matters**
- Ensures compatible environment for loading and running the model/UI.

## 4) End-to-End Inference Flow

1. User runs Streamlit app:
   - `streamlit run app.py`
2. `app.py` renders UI and presets.
3. User edits inputs and submits form.
4. `app.py` calls `predict_weather(inputs)` in `app/predictor.py`.
5. `predictor.py`:
   - validates and orders input features,
   - gets cached model from `app/model_loader.py`.
6. `model_loader.py` loads model path from `app/config.py`.
7. Model predicts class and probability.
8. Results are returned to UI and displayed.

## 5) End-to-End Training-Rebuild Flow

1. User runs:
   - `python3 scripts/rebuild_knn_training.py`
2. Script loads source model (`KNN_best_model_clean.pkl`).
3. Script extracts internal training arrays from source model.
4. Script creates and fits a new KNN with same params.
5. Script saves rebuilt model + extracted training CSV.
6. Script prints metrics and verifies identical predictions.

## 6) How to Run

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Streamlit UI

```bash
source .venv/bin/activate
streamlit run app.py
```

### Run training-rebuild script

```bash
source .venv/bin/activate
python3 scripts/rebuild_knn_training.py
```

## 7) Notes

- The predictor requires these exact 8 feature names and order (enforced in code).
- `scripts/rebuild_knn_training.py` is a recovered training-equivalent path, not the original historical training script.
- If `models/KNN_best_model_clean.pkl` is missing, both prediction and rebuild workflows will fail.
