from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent

RUN_DIR = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "finetuning_runs"
    / "yolov8n_200_images_lr_0.0005_epochs_40"
)

RESULTS_CSV = RUN_DIR / "results.csv"

OUTPUT_PATH = RUN_DIR / "training_progress_log.png"


def main():
    df = pd.read_csv(RESULTS_CSV)

    # Strip spaces from column names
    df.columns = [
        column.strip()
        for column in df.columns
    ]

    epochs = df["epoch"]

    # YOLO has 3 separate losses.
    # Add them together to give us one
    # overall train and validation loss.
    train_loss = (
        df["train/box_loss"]
        + df["train/cls_loss"]
        + df["train/dfl_loss"]
    )

    val_loss = (
        df["val/box_loss"]
        + df["val/cls_loss"]
        + df["val/dfl_loss"]
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss",
    )

    # Mahdi specifically wanted a log scale
    plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")

    plt.title(
        "YOLOv8n Fine-Tuning Progress\n"
        "Learning Rate = 0.0005"
    )

    plt.legend()
    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
    )

    print(
        f"Saved graph to:\n"
        f"{OUTPUT_PATH}"
    )

    plt.show()


if __name__ == "__main__":
    main()