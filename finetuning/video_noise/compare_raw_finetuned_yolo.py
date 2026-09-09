from pathlib import Path
import csv

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

VIDEO_DIR = (
    PROJECT_DIR
    / "finetuning_dataset"
    / "video_clips"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "finetuning_results"
    / "raw_vs_finetuned"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINETUNED_MODEL = (
    PROJECT_DIR
    / "finetuning_runs"
    / "experiments"
    / "lr_0.0005_epochs_40"
    / "weights"
    / "best.pt"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
}

CONFIDENCE = 0.25


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading raw YOLOv8n...")
raw_model = YOLO("yolov8n.pt")

print("Loading fine-tuned YOLOv8n...")
finetuned_model = YOLO(
    str(FINETUNED_MODEL)
)


# --------------------------------------------------
# Detection
# --------------------------------------------------

def count_detections(
    model,
    frame,
):
    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False,
    )

    result = results[0]

    counts = {
        "person": 0,
        "vehicle": 0,
        "cat": 0,
        "dog": 0,
    }

    for box in result.boxes:
        class_id = int(
            box.cls.item()
        )

        label = result.names[
            class_id
        ]

        if label == "person":
            counts["person"] += 1

        elif label in {
            "car",
            "truck",
            "bus",
            "motorcycle",
        }:
            counts["vehicle"] += 1

        elif label == "cat":
            counts["cat"] += 1

        elif label == "dog":
            counts["dog"] += 1

        # Fine-tuned model uses the combined
        # "vehicle" class directly.
        elif label == "vehicle":
            counts["vehicle"] += 1

    return counts, result


# --------------------------------------------------
# Analyse one video
# --------------------------------------------------

def analyse_video(video_path):
    print()
    print("=" * 60)
    print(f"VIDEO: {video_path.name}")
    print("=" * 60)

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")

    rows = []

    frame_number = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        raw_counts, _ = count_detections(
            raw_model,
            frame,
        )

        fine_counts, _ = count_detections(
            finetuned_model,
            frame,
        )

        raw_total = sum(
            raw_counts.values()
        )

        fine_total = sum(
            fine_counts.values()
        )

        timestamp = (
            frame_number / fps
            if fps > 0
            else 0
        )

        rows.append(
            {
                "frame": frame_number,
                "time_seconds": round(
                    timestamp,
                    3,
                ),
                "raw_person":
                    raw_counts["person"],
                "finetuned_person":
                    fine_counts["person"],
                "raw_vehicle":
                    raw_counts["vehicle"],
                "finetuned_vehicle":
                    fine_counts["vehicle"],
                "raw_cat":
                    raw_counts["cat"],
                "finetuned_cat":
                    fine_counts["cat"],
                "raw_dog":
                    raw_counts["dog"],
                "finetuned_dog":
                    fine_counts["dog"],
                "raw_total":
                    raw_total,
                "finetuned_total":
                    fine_total,
            }
        )

        print(
            f"Frame {frame_number}: "
            f"raw={raw_total} | "
            f"fine-tuned={fine_total}"
        )

        frame_number += 1

    capture.release()

    return rows


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

def save_csv(
    video_name,
    rows,
):
    output_path = (
        OUTPUT_DIR
        / f"{video_name}_comparison.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved CSV: {output_path}"
    )


# --------------------------------------------------
# Comparison graph
# --------------------------------------------------

def save_graph(
    video_name,
    rows,
):
    frames = [
        row["frame"]
        for row in rows
    ]

    raw = [
        row["raw_total"]
        for row in rows
    ]

    fine = [
        row["finetuned_total"]
        for row in rows
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        frames,
        raw,
        label="Raw YOLOv8n",
    )

    plt.plot(
        frames,
        fine,
        label="Fine-Tuned YOLOv8n",
    )

    plt.xlabel(
        "Frame Number"
    )

    plt.ylabel(
        "Total Detections"
    )

    plt.title(
        "Raw vs Fine-Tuned YOLOv8n\n"
        f"{video_name}"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{video_name}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved graph: {output_path}"
    )


# --------------------------------------------------
# Create side-by-side annotated video
# --------------------------------------------------

def create_side_by_side_video(
    video_path,
):
    print()
    print(
        f"Creating annotated comparison: "
        f"{video_path.name}"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    output_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_side_by_side.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width * 2, height),
    )

    if not writer.isOpened():
        capture.release()

        raise ValueError(
            f"Could not create output video: "
            f"{output_path}"
        )

    frame_number = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        # Run raw YOLO
        raw_counts, raw_result = (
            count_detections(
                raw_model,
                frame,
            )
        )

        # Run fine-tuned YOLO
        fine_counts, fine_result = (
            count_detections(
                finetuned_model,
                frame,
            )
        )

        # Draw YOLO bounding boxes,
        # class labels and confidence scores.
        raw_frame = raw_result.plot()
        fine_frame = fine_result.plot()

        raw_total = sum(
            raw_counts.values()
        )

        fine_total = sum(
            fine_counts.values()
        )

        # --------------------------------------------------
        # Headings
        # --------------------------------------------------

        cv2.putText(
            raw_frame,
            "RAW YOLOv8n",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            fine_frame,
            "FINE-TUNED YOLOv8n",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------
        # Detection counts
        # --------------------------------------------------

        cv2.putText(
            raw_frame,
            f"Relevant detections: {raw_total}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            fine_frame,
            f"Relevant detections: {fine_total}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------
        # Frame number
        # --------------------------------------------------

        cv2.putText(
            raw_frame,
            f"Frame: {frame_number}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            fine_frame,
            f"Frame: {frame_number}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Put both versions next to each other.
        comparison_frame = cv2.hconcat(
            [
                raw_frame,
                fine_frame,
            ]
        )

        writer.write(
            comparison_frame
        )

        frame_number += 1

    capture.release()
    writer.release()

    print(
        f"Saved video: {output_path}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not FINETUNED_MODEL.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found:\n"
            f"{FINETUNED_MODEL}"
        )

    videos = sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if path.suffix.lower()
        in VIDEO_EXTENSIONS
    )

    print()
    print(
        f"Found {len(videos)} videos."
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE}"
    )

    for video_path in videos:

        rows = analyse_video(
            video_path
        )

        if not rows:
            continue

        save_csv(
            video_path.stem,
            rows,
        )

        save_graph(
            video_path.stem,
            rows,
        )

        create_side_by_side_video(
            video_path
        )

    print()
    print("=" * 60)
    print(
        "RAW VS FINE-TUNED "
        "COMPARISON COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()