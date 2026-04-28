from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "KNN_best_model_clean.pkl"


def resolve_model_path() -> Path:
	if MODEL_PATH.exists():
		return MODEL_PATH
	raise FileNotFoundError(
		f"Could not find the required KNN model file: {MODEL_PATH}"
	)