from pathlib import Path
import json
import sys
import time

import cv2
import torch

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "evaluation_dataset"
RESULTS_DIR = BASE_DIR / "results" / "yolox_x"

YOLOX_DIR = PROJECT_DIR / "YOLOX"
WEIGHTS_PATH = YOLOX_DIR / "weights" / "yolox_x.pth"

# Let Python import YOLOX from the cloned repo
sys.path.insert(0, str(YOLOX_DIR))

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess


MODEL_NAME = "yolox-x"

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3

# COCO person class
PERSON_CLASS_ID = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_images():
    return sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_model():
    exp = get_exp(None, MODEL_NAME)

    model = exp.get_model()
    model.eval()

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    return model, exp


def preprocess(image, test_size):
    transform = ValTransform(
        legacy=False
    )

    processed, _ = transform(
        image,
        None,
        test_size,
    )

    tensor = torch.from_numpy(
        processed
    ).unsqueeze(0)

    tensor = tensor.float()

    return tensor


def run_inference(
    model,
    exp,
    image_path,
):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    ratio = min(
        exp.test_size[0] / height,
        exp.test_size[1] / width,
    )

    tensor = preprocess(
        image,
        exp.test_size,
    )

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(tensor)

        outputs = postprocess(
            outputs,
            exp.num_classes,
            CONF_THRESHOLD,
            NMS_THRESHOLD,
            class_agnostic=True,
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    detections = []

    output = outputs[0]

    if output is not None:
        output = output.cpu()

        boxes = output[:, 0:4]
        boxes /= ratio

        object_conf = output[:, 4]
        class_conf = output[:, 5]
        class_ids = output[:, 6]

        scores = (
            object_conf
            * class_conf
        )

        for box, score, class_id in zip(
            boxes,
            scores,
            class_ids,
        ):
            class_id = int(
                class_id.item()
            )

            if class_id != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = (
                box.tolist()
            )

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

    return (
        image,
        detections,
        elapsed,
    )


def draw_detections(
    image,
    detections,
):
    for detection in detections:
        box = detection["box"]

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        confidence = (
            detection["confidence"]
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"person {confidence:.2f}"
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return image


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing weights: {WEIGHTS_PATH}"
        )

    images = get_images()

    print(
        f"Found {len(images)} images."
    )

    model, exp = prepare_model()

    annotated_dir = (
        RESULTS_DIR / "annotated"
    )

    json_dir = (
        RESULTS_DIR / "json"
    )

    annotated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    total_time = 0.0

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        category = (
            image_path.parent.name
        )

        print(
            f"[{index}/{len(images)}] "
            f"{category}/{image_path.name}"
        )

        image, detections, elapsed = (
            run_inference(
                model,
                exp,
                image_path,
            )
        )

        total_time += elapsed

        result = {
            "image": image_path.name,
            "category": category,
            "model": "yolox_x",
            "person_count": len(
                detections
            ),
            "inference_seconds": round(
                elapsed,
                4,
            ),
            "detections": detections,
        }

        all_results.append(result)

        # Save JSON
        category_json_dir = (
            json_dir / category
        )

        category_json_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            category_json_dir
            / f"{image_path.stem}.json"
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
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

        annotated = draw_detections(
            image.copy(),
            detections,
        )

        output_path = (
            category_annotated_dir
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            annotated,
        )

        print(
            f"    People: "
            f"{len(detections)} | "
            f"Time: {elapsed:.3f}s"
        )

    combined_path = (
        RESULTS_DIR / "results.json"
    )

    with combined_path.open(
        "w",
        encoding="utf-8",
    ) as file:
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

    print("\nYOLOX-X complete.")
    print(
        f"Average time: "
        f"{average_time:.3f}s/image"
    )
    print(
        f"Results: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()