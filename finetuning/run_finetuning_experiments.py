from pathlib import Path
import csv

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_YAML = (
    PROJECT_DIR
    / "yolo_finetune"
    / "dataset.yaml"
)

RUNS_DIR = (
    PROJECT_DIR
    / "finetuning_runs"
    / "experiments"
)

MODEL_NAME = "yolov8n.pt"

LEARNING_RATES = [
    0.01,
    0.005,
    0.001,
    0.0005,
]

EPOCH_OPTIONS = [
    10,
    20,
    30,
    40,
]


def run_experiment(
    learning_rate,
    epochs,
):
    run_name = (
        f"lr_{learning_rate}_"
        f"epochs_{epochs}"
    )

    print()
    print("=" * 70)
    print(f"RUNNING: {run_name}")
    print("=" * 70)

    # IMPORTANT:
    # Start every experiment from the same
    # pretrained YOLOv8n weights.
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATASET_YAML),

        epochs=epochs,
        imgsz=640,
        batch=4,

        # Explicit optimizer so Ultralytics
        # does NOT override our learning rate.
        optimizer="AdamW",
        lr0=learning_rate,

        # Freeze early backbone layers.
        freeze=10,

        workers=2,

        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,

        seed=42,
        deterministic=True,
    )

    return {
        "run": run_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "mAP50": float(
            results.results_dict.get(
                "metrics/mAP50(B)",
                0,
            )
        ),
        "mAP50_95": float(
            results.results_dict.get(
                "metrics/mAP50-95(B)",
                0,
            )
        ),
        "precision": float(
            results.results_dict.get(
                "metrics/precision(B)",
                0,
            )
        ),
        "recall": float(
            results.results_dict.get(
                "metrics/recall(B)",
                0,
            )
        ),
    }


def save_summary(results):
    output_path = (
        RUNS_DIR
        / "experiment_summary.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "run",
        "learning_rate",
        "epochs",
        "mAP50",
        "mAP50_95",
        "precision",
        "recall",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"Summary saved to: "
        f"{output_path}"
    )


def main():
    all_results = []

    total_runs = (
        len(LEARNING_RATES)
        * len(EPOCH_OPTIONS)
    )

    current_run = 0

    for learning_rate in LEARNING_RATES:
        for epochs in EPOCH_OPTIONS:
            current_run += 1

            print()
            print(
                f"Experiment "
                f"{current_run}/{total_runs}"
            )

            result = run_experiment(
                learning_rate,
                epochs,
            )

            all_results.append(
                result
            )

            # Save after EVERY run so results
            # aren't lost if something stops.
            save_summary(
                all_results
            )

    print()
    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    best_result = max(
        all_results,
        key=lambda item: item[
            "mAP50_95"
        ],
    )

    print()
    print("Best configuration:")
    print(
        f"Learning rate: "
        f"{best_result['learning_rate']}"
    )
    print(
        f"Epochs: "
        f"{best_result['epochs']}"
    )
    print(
        f"mAP50: "
        f"{best_result['mAP50']:.4f}"
    )
    print(
        f"mAP50-95: "
        f"{best_result['mAP50_95']:.4f}"
    )
    print(
        f"Precision: "
        f"{best_result['precision']:.4f}"
    )
    print(
        f"Recall: "
        f"{best_result['recall']:.4f}"
    )


if __name__ == "__main__":
    main()