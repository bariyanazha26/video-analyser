from pathlib import Path
import json
import time
import cv2

from ultralytics import RTDETR


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "evaluation_dataset"
RESULTS_DIR = BASE_DIR / "results" / "rtdetr_l"

MODEL_WEIGHTS = "rtdetr-l.pt"

PERSON_CLASS = 0
CONF_THRESHOLD = 0.25

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_images():
    return sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    images = get_images()

    print(f"Found {len(images)} images.")

    model = RTDETR(MODEL_WEIGHTS)

    annotated_dir = RESULTS_DIR / "annotated"
    json_dir = RESULTS_DIR / "json"

    annotated_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_time = 0.0

    for index, image_path in enumerate(images, start=1):
        category = image_path.parent.name

        print(
            f"[{index}/{len(images)}] "
            f"{category}/{image_path.name}"
        )

        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            classes=[PERSON_CLASS],
            conf=CONF_THRESHOLD,
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
            "model": "rtdetr_l",
            "person_count": len(detections),
            "inference_seconds": round(elapsed, 4),
            "detections": detections,
        }

        all_results.append(image_result)

        category_json_dir = json_dir / category
        category_json_dir.mkdir(parents=True, exist_ok=True)

        with (
            category_json_dir / f"{image_path.stem}.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                image_result,
                file,
                indent=2,
            )

        category_annotated_dir = annotated_dir / category
        category_annotated_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        annotated_image = result.plot()

        cv2.imwrite(
            str(
                category_annotated_dir
                / image_path.name
            ),
            annotated_image,
        )

        print(
            f"    People: {len(detections)} | "
            f"Time: {elapsed:.3f}s"
        )

    with (
        RESULTS_DIR / "results.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=2,
        )

    average_time = (
        total_time / len(images)
        if images
        else 0
    )

    print("\nRT-DETR-L complete.")
    print(
        f"Average time: "
        f"{average_time:.3f}s/image"
    )
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()