import json
import time

import cv2
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

from models import (
    VIDEOS,
    RESOLUTIONS,
    RESULTS_DIR,
    ANNOTATED_DIR,
)


MODEL_NAME = "grounding_dino"
MODEL_ID = "IDEA-Research/grounding-dino-base"

TEXT_PROMPT = "person."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25


def draw_detections(frame, detections):
    for detection in detections:
        box = detection["box"]

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        confidence = detection["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"person {confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame


def detect_people(
    frame,
    processor,
    model,
    image_size,
):
    original_height, original_width = frame.shape[:2]

    resized_frame = cv2.resize(
        frame,
        (image_size, image_size),
    )

    rgb_frame = cv2.cvtColor(
        resized_frame,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(rgb_frame)

    inputs = processor(
        images=pil_image,
        text=TEXT_PROMPT,
        return_tensors="pt",
        do_resize=False,
    )

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([
        [image_size, image_size]
    ])

    processed = (
        processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=target_sizes,
        )[0]
    )

    elapsed = time.perf_counter() - start

    scale_x = original_width / image_size
    scale_y = original_height / image_size

    detections = []

    for box, score, label in zip(
        processed["boxes"],
        processed["scores"],
        processed["labels"],
    ):
        label_text = str(label).lower()

        if "person" not in label_text:
            continue

        x1, y1, x2, y2 = box.tolist()

        detections.append({
            "label": "person",
            "confidence": round(
                float(score),
                4,
            ),
            "box": {
                "x1": round(x1 * scale_x, 2),
                "y1": round(y1 * scale_y, 2),
                "x2": round(x2 * scale_x, 2),
                "y2": round(y2 * scale_y, 2),
            },
        })

    return detections, elapsed


def run_single_benchmark(
    processor,
    model,
    video_category,
    video_path,
    image_size,
):
    print("\n" + "=" * 70)
    print(
        f"Running {MODEL_NAME} | "
        f"{video_category} | "
        f"{image_size}"
    )
    print("=" * 70)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_folder = (
        ANNOTATED_DIR
        / MODEL_NAME
        / video_category
        / str(image_size)
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_folder = (
        RESULTS_DIR
        / MODEL_NAME
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

    video_fps = capture.get(
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
        video_fps,
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

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            detections, elapsed = detect_people(
                frame=frame,
                processor=processor,
                model=model,
                image_size=image_size,
            )

            person_count = len(detections)

            total_inference_time += elapsed
            total_detections += person_count

            annotated_frame = draw_detections(
                frame.copy(),
                detections,
            )

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
                frame_number % 25 == 0
                or frame_number == total_frames
            ):
                average_time = (
                    total_inference_time
                    / frame_number
                )

                print(
                    f"Progress: "
                    f"{frame_number}/{total_frames} | "
                    f"People: {person_count} | "
                    f"Avg time: "
                    f"{average_time:.3f}s/frame"
                )

    finally:
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
        "model": MODEL_NAME,
        "video": video_path.name,
        "category": video_category,
        "image_size": image_size,
        "video_fps": video_fps,
        "total_frames": frame_number,
        "total_person_detections": total_detections,
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
    print(f"Model:                {MODEL_NAME}")
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
    print("Loading Grounding DINO...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(MODEL_ID)
    )

    model.eval()

    total_runs = (
        len(VIDEOS)
        * len(RESOLUTIONS)
    )

    run_number = 0

    for video_category, video_path in VIDEOS.items():
        for image_size in RESOLUTIONS:
            run_number += 1

            print(
                f"\nRUN {run_number}/{total_runs}"
            )

            run_single_benchmark(
                processor=processor,
                model=model,
                video_category=video_category,
                video_path=video_path,
                image_size=image_size,
            )

    print("\n" + "=" * 70)
    print("ALL GROUNDING DINO BENCHMARKS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()