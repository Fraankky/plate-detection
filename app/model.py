from pathlib import Path
from typing import Any

from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "weights" / "best.pt"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model plate tidak ditemukan: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
print(f"Loaded model: {MODEL_PATH}")


def predict(image_np: Any, conf: float = 0.25, verbose: bool = False) -> Any:
    return list(model(image_np, conf=conf, verbose=verbose))[0]
