from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR.parent / "evaluation_dataset"
RESULTS_DIR = BASE_DIR / "results"

MODEL_NAME = "yolo11x.pt"

def run_yolo():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_DIR}"
        )

    image_files = list(DATASET_DIR.glob("*.jpg"))

    if not image_files:
        raise FileNotFoundError(
            f"No JPG images found in: {DATASET_DIR}"
        )

    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"Running model on {len(image_files)} images...")

    model.predict(
        source=str(DATASET_DIR),
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(RESULTS_DIR),
        name=Path(MODEL_NAME).stem,
        exist_ok=True
    )

    print("YOLO evaluation completed.")
    print(f"Results saved to: {RESULTS_DIR / Path(MODEL_NAME).stem}")


if __name__ == "__main__":
    run_yolo()