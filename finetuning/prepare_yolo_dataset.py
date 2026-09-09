from pathlib import Path
import json
import random
import shutil


PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    PROJECT_DIR
    / "finetuning_dataset"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "yolo_finetune"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CLASS_MAP = {
    "person": 0,
    "vehicle": 1,
    "cat": 2,
    "dog": 3,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

SOURCE_FOLDERS = [
    "people",
    "animals",
    "negative",
    "additional",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# --------------------------------------------------
# FIND LABELLED IMAGES
# --------------------------------------------------

def get_dataset_items():
    items = []

    for folder_name in SOURCE_FOLDERS:
        folder = SOURCE_DIR / folder_name

        if not folder.exists():
            print(
                f"Skipping missing folder: "
                f"{folder}"
            )
            continue

        for image_path in sorted(
            folder.iterdir()
        ):
            if (
                image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            json_path = (
                image_path.with_suffix(
                    ".json"
                )
            )

            if not json_path.exists():
                print(
                    f"Including {image_path.name}: "
                    f"no JSON -> negative image"
                )
                json_path = None

            items.append(
                {
                    "image": image_path,
                    "json": json_path,
                }
            )

    return items


# --------------------------------------------------
# LABELME → YOLO CONVERSION
# --------------------------------------------------

def convert_labelme_to_yolo(
    json_path,
):
    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    image_width = data[
        "imageWidth"
    ]

    image_height = data[
        "imageHeight"
    ]

    yolo_lines = []

    for shape in data.get(
        "shapes",
        [],
    ):
        label = (
            shape.get(
                "label",
                "",
            )
            .strip()
            .lower()
        )

        if label == "__ignore__":
            continue

        if label not in CLASS_MAP:
            print(
                f"Warning: ignoring "
                f"unknown label "
                f"'{label}' in "
                f"{json_path.name}"
            )
            continue

        if (
            shape.get(
                "shape_type"
            )
            != "rectangle"
        ):
            print(
                f"Warning: ignoring "
                f"non-rectangle shape "
                f"in {json_path.name}"
            )
            continue

        points = shape.get(
            "points",
            [],
        )

        if len(points) != 2:
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]

        left = min(x1, x2)
        right = max(x1, x2)

        top = min(y1, y2)
        bottom = max(y1, y2)

        box_width = (
            right - left
        )

        box_height = (
            bottom - top
        )

        x_center = (
            left
            + box_width / 2
        )

        y_center = (
            top
            + box_height / 2
        )

        # Normalise to 0–1
        x_center /= image_width
        y_center /= image_height

        box_width /= image_width
        box_height /= image_height

        class_id = CLASS_MAP[
            label
        ]

        yolo_lines.append(
            (
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )
        )

    return yolo_lines


# --------------------------------------------------
# SPLIT DATASET
# --------------------------------------------------

def split_dataset(items):
    random.seed(
        RANDOM_SEED
    )

    shuffled = items.copy()

    random.shuffle(
        shuffled
    )

    total = len(shuffled)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    train_items = shuffled[
        :train_count
    ]

    val_items = shuffled[
        train_count:
        train_count + val_count
    ]

    test_items = shuffled[
        train_count + val_count:
    ]

    return {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }


# --------------------------------------------------
# CREATE OUTPUT FOLDERS
# --------------------------------------------------

def create_output_folders():
    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    for split in [
        "train",
        "val",
        "test",
    ]:
        (
            OUTPUT_DIR
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_DIR
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# --------------------------------------------------
# COPY IMAGE + CREATE YOLO LABEL
# --------------------------------------------------

def prepare_split(
    split_name,
    items,
):
    images_dir = (
        OUTPUT_DIR
        / "images"
        / split_name
    )

    labels_dir = (
        OUTPUT_DIR
        / "labels"
        / split_name
    )

    for item in items:
        image_path = item[
            "image"
        ]

        json_path = item[
            "json"
        ]

        target_image = (
            images_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            target_image,
        )

        if json_path is None:
            # Negative image: no target objects
            yolo_lines = []
        else:
            yolo_lines = (
                convert_labelme_to_yolo(
                    json_path
                )
            )

        target_label = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        with target_label.open(
            "w",
            encoding="utf-8",
        ) as file:
            if yolo_lines:
                file.write(
                    "\n".join(
                        yolo_lines
                    )
                    + "\n"
                )

        print(
            f"{split_name:<5} "
            f"{image_path.name}: "
            f"{len(yolo_lines)} boxes"
        )


# --------------------------------------------------
# DATASET YAML
# --------------------------------------------------

def create_dataset_yaml():
    yaml_path = (
        OUTPUT_DIR
        / "dataset.yaml"
    )

    yaml_text = f"""path: {OUTPUT_DIR}

train: images/train
val: images/val
test: images/test

names:
  0: person
  1: vehicle
  2: cat
  3: dog
"""

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )

    return yaml_path


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    items = (
        get_dataset_items()
    )

    if not items:
        raise FileNotFoundError(
            "No labelled images found."
        )

    print(
        f"Found {len(items)} "
        f"labelled images."
    )

    create_output_folders()

    splits = split_dataset(
        items
    )

    print()
    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    for split_name, split_items in (
        splits.items()
    ):
        print(
            f"{split_name:<5}: "
            f"{len(split_items)}"
        )

    print()

    for split_name, split_items in (
        splits.items()
    ):
        prepare_split(
            split_name,
            split_items,
        )

    yaml_path = (
        create_dataset_yaml()
    )

    print()
    print("=" * 60)
    print("YOLO DATASET READY")
    print("=" * 60)

    print(
        f"Output: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"YAML:   "
        f"{yaml_path}"
    )


if __name__ == "__main__":
    main()