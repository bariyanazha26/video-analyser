from pathlib import Path
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "finetuning_dataset"


DATASETS = {
    "people": {
        "coco_json": DATASET_DIR / "people_annotations.json",
        "output_dir": DATASET_DIR / "people",
    },
    "animals": {
        "coco_json": DATASET_DIR / "animals_annotations.json",
        "output_dir": DATASET_DIR / "animals",
    },
    "negative": {
        "coco_json": DATASET_DIR / "negative_annotations.json",
        "output_dir": DATASET_DIR / "negative",
    },
}


def convert_dataset(name, coco_json, output_dir):
    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    if not coco_json.exists():
        print(f"Missing COCO file: {coco_json}")
        return 0

    if not output_dir.exists():
        print(f"Missing output folder: {output_dir}")
        return 0

    with coco_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    categories = {
        category["id"]: category["name"].strip().lower()
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
        image_name = Path(
            image_info["file_name"]
        ).name

        image_path = output_dir / image_name
        json_path = image_path.with_suffix(".json")

        if not image_path.exists():
            print(
                f"Warning: image not found: "
                f"{image_name}"
            )

        shapes = []

        for annotation in annotations_by_image.get(
            image_id,
            [],
        ):
            category_id = annotation["category_id"]

            if category_id not in categories:
                print(
                    f"Warning: unknown category ID "
                    f"{category_id} in {image_name}"
                )
                continue

            category_name = categories[category_id]

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
        f"{name}: "
        f"{converted_count} JSON files created."
    )

    return converted_count


def main():
    total_converted = 0

    for name, config in DATASETS.items():
        total_converted += convert_dataset(
            name=name,
            coco_json=config["coco_json"],
            output_dir=config["output_dir"],
        )

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    print(
        f"Total JSON files created: "
        f"{total_converted}"
    )


if __name__ == "__main__":
    main()