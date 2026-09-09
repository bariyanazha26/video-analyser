from pathlib import Path
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "finetuning_dataset"

VALID_LABELS = {
    "person",
    "cat",
    "vehicle",
    "dog",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def get_images(folder):
    return sorted(
        path
        for path in folder.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_folder(folder_name):
    folder = DATASET_DIR / folder_name

    if not folder.exists():
        print(f"\nMissing folder: {folder}")
        return {
            "images": 0,
            "missing_json": 0,
            "invalid_json": 0,
            "invalid_labels": 0,
        }

    images = get_images(folder)

    missing_json = []
    invalid_json = []
    invalid_labels = []

    for image_path in images:
        json_path = image_path.with_suffix(".json")

        if not json_path.exists():
            missing_json.append(image_path.name)
            continue

        try:
            with json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (json.JSONDecodeError, OSError):
            invalid_json.append(json_path.name)
            continue

        shapes = data.get("shapes", [])

        for shape in shapes:
            label = (
                shape.get("label", "")
                .strip()
                .lower()
            )

            # Ignore LabelMe ignore regions
            if label == "__ignore__":
                continue

            if label not in VALID_LABELS:
                invalid_labels.append(
                    {
                        "file": json_path.name,
                        "label": label,
                    }
                )

    print("\n" + "=" * 60)
    print(folder_name.upper())
    print("=" * 60)

    print(f"Images:           {len(images)}")
    print(f"Missing JSON:     {len(missing_json)}")
    print(f"Invalid JSON:     {len(invalid_json)}")
    print(f"Invalid labels:   {len(invalid_labels)}")

    if missing_json:
        print("\nMissing JSON files:")
        for filename in missing_json:
            print(f"  - {filename}")

    if invalid_json:
        print("\nInvalid JSON files:")
        for filename in invalid_json:
            print(f"  - {filename}")

    if invalid_labels:
        print("\nInvalid labels:")
        for item in invalid_labels:
            print(
                f"  - {item['file']}: {item['label']}"
            )

    return {
        "images": len(images),
        "missing_json": len(missing_json),
        "invalid_json": len(invalid_json),
        "invalid_labels": len(invalid_labels),
    }


def main():
    categories = [
        "people",
        "animals",
        "negative",
        "additional",
    ]

    totals = {
        "images": 0,
        "missing_json": 0,
        "invalid_json": 0,
        "invalid_labels": 0,
    }

    for category in categories:
        result = validate_folder(category)

        for key in totals:
            totals[key] += result[key]

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Total images:     {totals['images']}")
    print(f"Missing JSON:     {totals['missing_json']}")
    print(f"Invalid JSON:     {totals['invalid_json']}")
    print(f"Invalid labels:   {totals['invalid_labels']}")

    is_valid = (
        totals["missing_json"] == 0
        and totals["invalid_json"] == 0
        and totals["invalid_labels"] == 0
    )

    print()

    if is_valid:
        print("Dataset valid ✅")
    else:
        print("Dataset has issues ❌")


if __name__ == "__main__":
    main()