from pathlib import Path
import json
import time

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "evaluation_dataset"
RESULTS_DIR = BASE_DIR / "results"

MODELS = {
    "yolo11m": "yolo11m.pt",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# COCO class 0 = person
PERSON_CLASS = 0

CONFIDENCE_THRESHOLD = 0.25


def get_images():
    images = [
        path
        for path in DATASET_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)


def run_model(model_name, weights, images):
    print(f"\n{'=' * 50}")
    print(f"Running {model_name}")
    print(f"{'=' * 50}")

    model = YOLO(weights)

    model_output_dir = RESULTS_DIR / model_name
    annotated_dir = model_output_dir / "annotated"
    json_dir = model_output_dir / "json"

    annotated_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_time = 0

    for index, image_path in enumerate(images, start=1):

        # crowded / sparse / normal / difficult_negative
        category = image_path.parent.name

        print(
            f"[{index}/{len(images)}] "
            f"{category}/{image_path.name}"
        )

        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            classes=[PERSON_CLASS],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        elapsed = time.perf_counter() - start
        total_time += elapsed

        result = results[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:

                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0])

                detections.append({
                    "label": "person",
                    "confidence": round(confidence, 4),
                    "box": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2),
                    },
                })

        image_result = {
            "image": image_path.name,
            "category": category,
            "model": model_name,
            "person_count": len(detections),
            "inference_seconds": round(elapsed, 4),
            "detections": detections,
        }

        all_results.append(image_result)

        # Save JSON
        category_json_dir = json_dir / category
        category_json_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            category_json_dir /
            f"{image_path.stem}.json"
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                image_result,
                file,
                indent=2,
            )

        # Save annotated image
        category_annotated_dir = (
            annotated_dir / category
        )

        category_annotated_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        annotated_image = result.plot()

        output_image_path = (
            category_annotated_dir /
            image_path.name
        )

        import cv2

        cv2.imwrite(
            str(output_image_path),
            annotated_image,
        )

        print(
            f"    People: {len(detections)} | "
            f"Time: {elapsed:.3f}s"
        )

    combined_path = model_output_dir / "results.json"

    with combined_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=2,
        )

    average_time = total_time / len(images)

    print(f"\n{model_name} complete")
    print(f"Average time: {average_time:.3f}s/image")
    print(f"Results: {model_output_dir}")


def main():
    images = get_images()

    print(f"Found {len(images)} evaluation images.")

    if len(images) != 20:
        print(
            f"Warning: expected 20 images, found {len(images)}."
        )

    for model_name, weights in MODELS.items():
        run_model(
            model_name,
            weights,
            images,
        )

    print("\nAll YOLOv8 models complete.")


if __name__ == "__main__":
    main()