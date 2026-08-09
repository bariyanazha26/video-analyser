from pathlib import Path
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent

COCO_JSON = (
    PROJECT_DIR
    / "finetuning_dataset"
    / "instances_Train.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "finetuning_dataset"
    / "people"
)


def main():
    with COCO_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    categories = {
        category["id"]: category["name"]
        for category in coco["categories"]
    }

    images = {
        image["id"]: image
        for image in coco["images"]
    }

    annotations_by_image = {}

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            [],
        ).append(annotation)

    converted_count = 0

    for image_id, image_info in images.items():
        image_name = image_info["file_name"]

        image_path = OUTPUT_DIR / image_name
        json_path = image_path.with_suffix(".json")

        shapes = []

        for annotation in annotations_by_image.get(
            image_id,
            [],
        ):
            category_name = categories[
                annotation["category_id"]
            ]

            x, y, width, height = annotation["bbox"]

            x2 = x + width
            y2 = y + height

            shapes.append(
                {
                    "label": category_name,
                    "points": [
                        [
                            round(x, 2),
                            round(y, 2),
                        ],
                        [
                            round(x2, 2),
                            round(y2, 2),
                        ],
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                }
            )

        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": image_info["height"],
            "imageWidth": image_info["width"],
        }

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                labelme_data,
                file,
                indent=2,
            )

        converted_count += 1

        print(
            f"Created: {json_path.name} "
            f"({len(shapes)} annotations)"
        )

    print()
    print(
        f"Conversion complete. "
        f"{converted_count} JSON files created."
    )


if __name__ == "__main__":
    main()