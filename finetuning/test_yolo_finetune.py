from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_YAML = (
    PROJECT_DIR
    / "yolo_finetune"
    / "dataset.yaml"
)

MODEL_NAME = "yolov8n.pt"


def main():
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATASET_YAML),

        # Small test only
        epochs=2,
        imgsz=640,
        batch=4,

        # Learning rate
        lr0=0.01,

        # Freeze backbone layers
        freeze=10,

        # Keep test quick
        workers=2,

        project=str(
            PROJECT_DIR
            / "finetuning_runs"
        ),

        name="test_run",

        exist_ok=True,
    )


if __name__ == "__main__":
    main()