import json
import time

import cv2
from ultralytics import YOLO

from models import (
    VIDEOS,
    MODELS,
    RESOLUTIONS,
    RESULTS_DIR,
    ANNOTATED_DIR,
    CONF_THRESHOLD,
)


def run_single_benchmark(
    model_name,
    model_weights,
    video_category,
    video_path,
    image_size,
):
    print("\n" + "=" * 70)
    print(
        f"Running {model_name} | "
        f"{video_category} | "
        f"{image_size}"
    )
    print("=" * 70)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    model = YOLO(model_weights)

    output_folder = (
        ANNOTATED_DIR
        / model_name
        / video_category
        / str(image_size)
    )
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_folder = (
        RESULTS_DIR
        / model_name
        / video_category
        / str(image_size)
    )
    result_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video_path = (
        output_folder
        / f"{video_path.stem}_annotated.mp4"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
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

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: "
            f"{output_video_path}"
        )

    frame_number = 0
    total_inference_time = 0.0
    total_detections = 0
    frame_results = []

    while True:
        success, frame = capture.read()

        if not success:
            break

        start = time.perf_counter()

        results = model.predict(
            source=frame,
            classes=[0],
            conf=CONF_THRESHOLD,
            imgsz=image_size,
            verbose=False,
        )

        elapsed = (
            time.perf_counter() - start
        )

        total_inference_time += elapsed

        result = results[0]

        person_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        total_detections += person_count

        annotated_frame = result.plot()
        writer.write(annotated_frame)

        frame_results.append({
            "frame": frame_number,
            "person_count": person_count,
            "inference_seconds": round(
                elapsed,
                4,
            ),
        })

        frame_number += 1

        if (
            frame_number % 50 == 0
            or frame_number == total_frames
        ):
            print(
                f"Progress: "
                f"{frame_number}/{total_frames} | "
                f"People: {person_count} | "
                f"Avg time: "
                f"{total_inference_time / frame_number:.3f}s/frame"
            )

    capture.release()
    writer.release()

    average_inference_time = (
        total_inference_time / frame_number
        if frame_number > 0
        else 0
    )

    processing_fps = (
        1 / average_inference_time
        if average_inference_time > 0
        else 0
    )

    average_detections = (
        total_detections / frame_number
        if frame_number > 0
        else 0
    )

    summary = {
        "model": model_name,
        "video": video_path.name,
        "category": video_category,
        "image_size": image_size,
        "video_fps": fps,
        "total_frames": frame_number,
        "total_person_detections": (
            total_detections
        ),
        "average_detections_per_frame": round(
            average_detections,
            4,
        ),
        "average_inference_seconds": round(
            average_inference_time,
            4,
        ),
        "processing_fps": round(
            processing_fps,
            2,
        ),
        "frames": frame_results,
    }

    results_path = (
        result_folder / "results.json"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nBenchmark complete.")
    print(f"Model:                {model_name}")
    print(f"Video:                {video_path.name}")
    print(f"Category:             {video_category}")
    print(f"Resolution:           {image_size}")
    print(f"Frames processed:     {frame_number}")
    print(f"Total detections:     {total_detections}")
    print(
        f"Avg detections/frame: "
        f"{average_detections:.2f}"
    )
    print(
        f"Avg inference time:   "
        f"{average_inference_time:.3f}s/frame"
    )
    print(
        f"Processing speed:     "
        f"{processing_fps:.2f} FPS"
    )
    print(
        f"Annotated video:      "
        f"{output_video_path}"
    )
    print(
        f"Results JSON:         "
        f"{results_path}"
    )


def main():
    total_runs = (
        len(MODELS)
        * len(VIDEOS)
        * len(RESOLUTIONS)
    )

    run_number = 0

    for model_name, model_info in MODELS.items():
        model_weights = model_info["weights"]

        for video_category, video_path in VIDEOS.items():
            for image_size in RESOLUTIONS:
                run_number += 1

                print(
                    f"\nRUN {run_number}/{total_runs}"
                )

                run_single_benchmark(
                    model_name=model_name,
                    model_weights=model_weights,
                    video_category=video_category,
                    video_path=video_path,
                    image_size=image_size,
                )

    print("\n" + "=" * 70)
    print("ALL VIDEO BENCHMARKS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()