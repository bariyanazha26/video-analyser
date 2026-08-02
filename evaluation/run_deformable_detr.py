from pathlib import Path
import json
import time

import cv2
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "evaluation_dataset"
RESULTS_DIR = BASE_DIR / "results" / "deformable_detr"

MODEL_ID = "SenseTime/deformable-detr"

CONF_THRESHOLD = 0.25

# COCO person class
PERSON_CLASS_ID = 1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_images():
    return sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def draw_detections(image, detections):
    for detection in detections:
        box = detection["box"]

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        confidence = detection["confidence"]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"person {confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return image


def main():
    images = get_images()

    print(f"Found {len(images)} images.")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    model = DeformableDetrForObjectDetection.from_pretrained(
        MODEL_ID
    )

    model.eval()

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

        pil_image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=pil_image,
            return_tensors="pt",
        )

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([
            [pil_image.height, pil_image.width]
        ])

        results = processor.post_process_object_detection(
            outputs,
            threshold=CONF_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        elapsed = time.perf_counter() - start
        total_time += elapsed

        detections = []

        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
        ):
            if int(label) != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = box.tolist()

            detections.append({
                "label": "person",
                "confidence": round(
                    float(score),
                    4,
                ),
                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            })

        image_result = {
            "image": image_path.name,
            "category": category,
            "model": "deformable_detr",
            "person_count": len(detections),
            "inference_seconds": round(
                elapsed,
                4,
            ),
            "detections": detections,
        }

        all_results.append(image_result)

        category_json_dir = json_dir / category
        category_json_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            category_json_dir / f"{image_path.stem}.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                image_result,
                file,
                indent=2,
            )

        cv_image = cv2.imread(str(image_path))

        annotated = draw_detections(
            cv_image,
            detections,
        )

        category_annotated_dir = (
            annotated_dir / category
        )

        category_annotated_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(
                category_annotated_dir
                / image_path.name
            ),
            annotated,
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

    print("\nDeformable DETR complete.")
    print(
        f"Average time: "
        f"{average_time:.3f}s/image"
    )
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()