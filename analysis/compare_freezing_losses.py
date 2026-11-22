from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Get the main video-analyser project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder where graphs will be saved
OUTPUT_DIR = PROJECT_ROOT / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training results for the four freezing experiments
experiments = {
    "A: Backbone Frozen": PROJECT_ROOT / "runs/detect/runs/detect/finetuning_runs/yolov8l_594_clean_frozen_backbone/results.csv",
    "B: No Freezing": PROJECT_ROOT / "runs/detect/finetuning_runs/yolov8l_594_clean_no_freezing/results.csv",
    "C: Head Frozen": PROJECT_ROOT / "runs/detect/finetuning_runs/yolov8l_594_clean_frozen_head/results.csv",
    "D: Both Frozen": PROJECT_ROOT / "runs/detect/finetuning_runs/yolov8l_594_clean_frozen_both/results.csv"
}


def plot_losses(loss_type, filename, title):
    plt.figure(figsize=(11, 6))

    for name, path in experiments.items():
        # Read training results
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        # Calculate total loss
        columns = [
            f"{loss_type}/box_loss",
            f"{loss_type}/cls_loss",
            f"{loss_type}/dfl_loss"
        ]

        total_loss = df[columns].sum(axis=1)

        # Plot loss against epoch
        plt.plot(
            df["epoch"],
            total_loss,
            label=name,
            linewidth=2
        )

    # Graph formatting
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss (Box + Classification + DFL)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save graph inside analysis folder
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300)

    print(f"Graph saved to: {output_path}")

    # Display graph
    plt.show()


# Graph 1: Training Loss
plot_losses(
    "train",
    "freezing_training_loss.png",
    "YOLOv8L Freezing Experiments — Training Loss"
)

# Graph 2: Validation Loss
plot_losses(
    "val",
    "freezing_validation_loss.png",
    "YOLOv8L Freezing Experiments — Validation Loss"
)