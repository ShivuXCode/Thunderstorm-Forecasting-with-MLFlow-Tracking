from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CANDIDATES = [
	PROJECT_ROOT / "models" / "KNN_best_model_rebuilt.pkl",
	PROJECT_ROOT / "models" / "KNN_best_model_clean.pkl",
]


def resolve_model_path() -> Path:
	for model_path in MODEL_CANDIDATES:
		if model_path.exists():
			return model_path
	raise FileNotFoundError(
		"Could not find a usable KNN model file. Looked in: "
		+ ", ".join(str(path) for path in MODEL_CANDIDATES)
	)