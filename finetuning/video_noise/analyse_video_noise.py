from pathlib import Path
import csv

import cv2
import matplotlib.pyplot as plt

from finetuning.model_wrappers.dfine_wrapper import detect as dfine_detect
from finetuning.model_wrappers.detr_wrapper import detect as detr_detect
from finetuning.model_wrappers.yolox_wrapper import detect as yolox_detect
from finetuning.model_wrappers.yolo_wrapper import detect as yolo_detect


PROJECT_DIR = Path(__file__).resolve().parents[2]

VIDEO_DIR = (
    PROJECT_DIR
    / "finetuning_dataset"
    / "video_clips"
)

BASE_RESULTS_DIR = (
    PROJECT_DIR
    / "finetuning_results"
    / "video_noise"
)

BASE_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
}

MODELS = {
    "dfine_x": dfine_detect,
    "yolo_v8m": yolo_detect,
    "yolox_m": yolox_detect,
    "detr_resnet50": detr_detect,
}


def count_labels(predictions):
    counts = {
        "person": 0,
        "vehicle": 0,
        "cat": 0,
        "dog": 0,
    }

    for prediction in predictions:
        label = prediction["label"]

        if label in counts:
            counts[label] += 1

    return counts


def analyse_video(
    video_path,
    detector,
    model_results_dir,
):
    print("\n" + "=" * 60)
    print(video_path.name)
    print("=" * 60)

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(f"FPS:          {fps:.2f}")
    print(f"Total frames: {total_frames}")

    rows = []
    frame_index = 0

    temp_frame = (
        model_results_dir
        / "_temp_frame.jpg"
    )

    while True:
        success, frame = capture.read()

        if not success:
            break

        cv2.imwrite(
            str(temp_frame),
            frame,
        )

        predictions = detector(
            temp_frame
        )

        counts = count_labels(
            predictions
        )

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0
        )

        total_detections = sum(
            counts.values()
        )

        rows.append(
            {
                "frame": frame_index,
                "time_seconds": round(
                    timestamp,
                    3,
                ),
                "person": counts["person"],
                "vehicle": counts["vehicle"],
                "cat": counts["cat"],
                "dog": counts["dog"],
                "total": total_detections,
            }
        )

        print(
            f"Frame {frame_index}: "
            f"person={counts['person']} "
            f"vehicle={counts['vehicle']} "
            f"cat={counts['cat']} "
            f"dog={counts['dog']} "
            f"total={total_detections}"
        )

        frame_index += 1

    capture.release()

    if temp_frame.exists():
        temp_frame.unlink()

    return rows


def save_csv(
    video_name,
    rows,
    model_results_dir,
):
    output_path = (
        model_results_dir
        / f"{video_name}_counts.csv"
    )

    fieldnames = [
        "frame",
        "time_seconds",
        "person",
        "vehicle",
        "cat",
        "dog",
        "total",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def save_chart(
    video_name,
    rows,
    model_name,
    model_results_dir,
):
    frames = [
        row["frame"]
        for row in rows
    ]

    totals = [
        row["total"]
        for row in rows
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        frames,
        totals,
        marker="o",
        markersize=3,
    )

    plt.title(
        f"Detection Count per Frame - "
        f"{model_name} - {video_name}"
    )

    plt.xlabel(
        "Frame Number"
    )

    plt.ylabel(
        "Total Detections"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        model_results_dir
        / f"{video_name}_chart.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved chart: {output_path}")


def run_model(
    model_name,
    detector,
    videos,
):
    model_results_dir = (
        BASE_RESULTS_DIR
        / model_name
    )

    model_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n")
    print("#" * 60)
    print(f"MODEL: {model_name.upper()}")
    print("#" * 60)

    for video_path in videos:
        video_name = video_path.stem

        csv_path = (
            model_results_dir
            / f"{video_name}_counts.csv"
        )

        chart_path = (
            model_results_dir
            / f"{video_name}_chart.png"
        )

        # Skip clips that have already been completed.
        if (
            csv_path.exists()
            and chart_path.exists()
        ):
            print(
                f"Skipping {video_name}: "
                f"results already exist."
            )
            continue

        rows = analyse_video(
            video_path=video_path,
            detector=detector,
            model_results_dir=model_results_dir,
        )

        save_csv(
            video_name=video_name,
            rows=rows,
            model_results_dir=model_results_dir,
        )

        save_chart(
            video_name=video_name,
            rows=rows,
            model_name=model_name,
            model_results_dir=model_results_dir,
        )


def main():
    videos = sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if path.suffix.lower()
        in VIDEO_EXTENSIONS
    )

    print(
        f"Found {len(videos)} videos."
    )

    for model_name, detector in MODELS.items():
        run_model(
            model_name=model_name,
            detector=detector,
            videos=videos,
        )

    print()
    print("=" * 60)
    print("VIDEO NOISE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()