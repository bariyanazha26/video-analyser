from pathlib import Path
import json
import time

from PIL import Image, ImageDraw, ImageFont

from model_wrappers.yolo_wrapper import detect as yolo_detect
from model_wrappers.yolox_wrapper import detect as yolox_detect
from model_wrappers.detr_wrapper import detect as detr_detect
from model_wrappers.dfine_wrapper import detect as dfine_detect


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "finetuning_dataset"

RESULTS_DIR = PROJECT_DIR / "finetuning_results"
RESULTS_DIR.mkdir(exist_ok=True)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def get_images():
    images = []

    for category in [
        "people",
        "animals",
        "negative",
    ]:
        folder = DATASET_DIR / category

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(
                    {
                        "category": category,
                        "path": image_path,
                    }
                )

    return images


def save_results(model_name, results):
    output_path = (
        RESULTS_DIR
        / f"{model_name}_predictions.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(f"\nSaved JSON: {output_path}")


def save_annotated_image(
    image_path,
    predictions,
    model_name,
    category,
):
    output_dir = (
        RESULTS_DIR
        / model_name
        / category
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc",
            28,
        )
    except OSError:
        font = ImageFont.load_default()

    for prediction in predictions:
        label = prediction["label"]
        score = prediction["score"]

        x1, y1, x2, y2 = prediction["box"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=5,
        )

        text = f"{label} {score:.2f}"

        text_bbox = draw.textbbox(
            (x1, y1),
            text,
            font=font,
        )

        padding = 6

        text_width = (
            text_bbox[2]
            - text_bbox[0]
        )

        text_height = (
            text_bbox[3]
            - text_bbox[1]
        )

        label_x = x1
        label_y = max(
            0,
            y1 - text_height - (padding * 2),
        )

        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + text_width + (padding * 2),
                label_y + text_height + (padding * 2),
            ],
            fill="red",
        )

        draw.text(
            (
                label_x + padding,
                label_y + padding,
            ),
            text,
            fill="white",
            font=font,
        )

    output_path = (
        output_dir
        / image_path.name
    )

    annotated.save(
        output_path,
        quality=95,
    )


def run_model(model_name, detector):
    images = get_images()

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    results = []

    total_inference_time = 0

    for index, item in enumerate(
        images,
        start=1,
    ):
        image_path = item["path"]

        start_time = time.perf_counter()

        predictions = detector(
            image_path
        )

        inference_time = (
            time.perf_counter()
            - start_time
        )

        total_inference_time += inference_time

        save_annotated_image(
            image_path=image_path,
            predictions=predictions,
            model_name=model_name,
            category=item["category"],
        )

        results.append(
            {
                "image": image_path.name,
                "category": item["category"],
                "inference_seconds": round(
                    inference_time,
                    4,
                ),
                "predictions": predictions,
            }
        )

        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}: "
            f"{len(predictions)} detections "
            f"({inference_time:.3f}s)"
        )

    average_time = (
        total_inference_time / len(images)
        if images
        else 0
    )

    fps = (
        1 / average_time
        if average_time > 0
        else 0
    )

    print()
    print(f"Images:     {len(images)}")
    print(f"Avg time:   {average_time:.3f}s")
    print(f"FPS:        {fps:.2f}")

    save_results(
        model_name,
        results,
    )


def main():
    models = [
        (
            "yolo_v8m",
            yolo_detect,
        ),
        (
            "yolox_m",
            yolox_detect,
        ),
        (
            "detr_resnet50",
            detr_detect,
        ),
        (
            "dfine_x",
            dfine_detect,
        ),
    ]

    for model_name, detector in models:
        run_model(
            model_name,
            detector,
        )

    print("\n" + "=" * 60)
    print("ALL MODELS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()