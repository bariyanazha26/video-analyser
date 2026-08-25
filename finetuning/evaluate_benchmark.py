from pathlib import Path
import csv
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "finetuning_dataset"
RESULTS_DIR = PROJECT_DIR / "finetuning_results"

OUTPUT_JSON = RESULTS_DIR / "benchmark_summary.json"
OUTPUT_CSV = RESULTS_DIR / "benchmark_summary.csv"


MODELS = {
    "yolo_v8m": (
        RESULTS_DIR
        / "yolo_v8m_predictions.json"
    ),
    "yolox_m": (
        RESULTS_DIR
        / "yolox_m_predictions.json"
    ),
    "detr_resnet50": (
        RESULTS_DIR
        / "detr_resnet50_predictions.json"
    ),
    "dfine_x": (
        RESULTS_DIR
        / "dfine_x_predictions.json"
    ),
}


CATEGORIES = [
    "people",
    "animals",
    "negative",
]


IOU_THRESHOLD = 0.50


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def load_ground_truth():
    ground_truth = {}

    for category in CATEGORIES:
        folder = DATASET_DIR / category

        for json_path in sorted(
            folder.glob("*.json")
        ):
            with json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            image_name = data.get(
                "imagePath",
                json_path.with_suffix(".jpg").name,
            )

            objects = []

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

                if (
                    shape.get("shape_type")
                    != "rectangle"
                ):
                    continue

                points = shape.get(
                    "points",
                    [],
                )

                if len(points) != 2:
                    continue

                x1, y1 = points[0]
                x2, y2 = points[1]

                objects.append(
                    {
                        "label": label,
                        "box": [
                            min(x1, x2),
                            min(y1, y2),
                            max(x1, x2),
                            max(y1, y2),
                        ],
                    }
                )

            ground_truth[image_name] = {
                "category": category,
                "objects": objects,
            }

    return ground_truth


def load_predictions(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        rows = json.load(file)

    return {
        row["image"]: row
        for row in rows
    }


def count_by_label(objects):
    counts = {}

    for obj in objects:
        label = obj["label"]

        counts[label] = (
            counts.get(label, 0)
            + 1
        )

    return counts


def match_boxes(
    ground_truth_objects,
    predictions,
):
    matched_ious = []

    used_predictions = set()

    correct_matches = 0

    for gt in ground_truth_objects:
        best_iou = 0
        best_index = None

        for index, prediction in enumerate(
            predictions
        ):
            if index in used_predictions:
                continue

            if (
                prediction["label"]
                != gt["label"]
            ):
                continue

            iou = calculate_iou(
                gt["box"],
                prediction["box"],
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None:
            used_predictions.add(
                best_index
            )

            matched_ious.append(
                best_iou
            )

            if best_iou >= IOU_THRESHOLD:
                correct_matches += 1

    return {
        "matched_ious": matched_ious,
        "correct_matches": correct_matches,
    }


def evaluate_model(
    model_name,
    prediction_path,
    ground_truth,
):
    predictions = load_predictions(
        prediction_path
    )

    total_gt_objects = 0
    total_pred_objects = 0

    total_correct_matches = 0

    matched_ious = []

    exact_count_images = 0
    total_images = 0

    total_absolute_count_error = 0

    per_image_results = []

    for image_name, gt_data in ground_truth.items():
        total_images += 1

        gt_objects = gt_data["objects"]

        prediction_data = predictions.get(
            image_name,
            {
                "predictions": [],
            },
        )

        predicted_objects = (
            prediction_data.get(
                "predictions",
                [],
            )
        )

        total_gt_objects += len(
            gt_objects
        )

        total_pred_objects += len(
            predicted_objects
        )

        gt_counts = count_by_label(
            gt_objects
        )

        pred_counts = count_by_label(
            predicted_objects
        )

        all_labels = set(
            gt_counts
        ) | set(
            pred_counts
        )

        image_count_error = 0

        exact_counts = True

        for label in all_labels:
            gt_count = gt_counts.get(
                label,
                0,
            )

            pred_count = pred_counts.get(
                label,
                0,
            )

            difference = abs(
                gt_count
                - pred_count
            )

            image_count_error += (
                difference
            )

            if difference != 0:
                exact_counts = False

        if exact_counts:
            exact_count_images += 1

        total_absolute_count_error += (
            image_count_error
        )

        match_result = match_boxes(
            gt_objects,
            predicted_objects,
        )

        total_correct_matches += (
            match_result[
                "correct_matches"
            ]
        )

        matched_ious.extend(
            match_result[
                "matched_ious"
            ]
        )

        mean_image_iou = (
            sum(
                match_result[
                    "matched_ious"
                ]
            )
            / len(
                match_result[
                    "matched_ious"
                ]
            )
            if match_result[
                "matched_ious"
            ]
            else 0
        )

        per_image_results.append(
            {
                "image": image_name,
                "category": gt_data[
                    "category"
                ],
                "ground_truth_count": len(
                    gt_objects
                ),
                "prediction_count": len(
                    predicted_objects
                ),
                "count_error": (
                    image_count_error
                ),
                "exact_count": exact_counts,
                "mean_iou": round(
                    mean_image_iou,
                    4,
                ),
                "correct_matches": (
                    match_result[
                        "correct_matches"
                    ]
                ),
            }
        )

    mean_iou = (
        sum(matched_ious)
        / len(matched_ious)
        if matched_ious
        else 0
    )

    location_accuracy = (
        total_correct_matches
        / total_gt_objects
        if total_gt_objects
        else 0
    )

    exact_count_accuracy = (
        exact_count_images
        / total_images
        if total_images
        else 0
    )

    count_mae = (
        total_absolute_count_error
        / total_images
        if total_images
        else 0
    )

    return {
        "model": model_name,
        "images": total_images,
        "ground_truth_objects": (
            total_gt_objects
        ),
        "predicted_objects": (
            total_pred_objects
        ),
        "mean_matched_iou": round(
            mean_iou,
            4,
        ),
        "location_accuracy_at_iou_0_5": round(
            location_accuracy,
            4,
        ),
        "exact_count_accuracy": round(
            exact_count_accuracy,
            4,
        ),
        "count_mae": round(
            count_mae,
            4,
        ),
        "per_image": per_image_results,
    }


def save_json(results):
    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )


def save_csv(results):
    fieldnames = [
        "model",
        "images",
        "ground_truth_objects",
        "predicted_objects",
        "mean_matched_iou",
        "location_accuracy_at_iou_0_5",
        "exact_count_accuracy",
        "count_mae",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    key: result[key]
                    for key in fieldnames
                }
            )


def print_summary(results):
    print(
        "\nMODEL BENCHMARK SUMMARY"
    )
    print("=" * 105)

    print(
        f"{'Model':<18}"
        f"{'Mean IoU':>12}"
        f"{'Location Acc.':>18}"
        f"{'Count Acc.':>16}"
        f"{'Count MAE':>14}"
        f"{'Pred Objects':>16}"
    )

    print("-" * 105)

    for result in results:
        print(
            f"{result['model']:<18}"
            f"{result['mean_matched_iou']:>12.3f}"
            f"{result['location_accuracy_at_iou_0_5']:>18.2%}"
            f"{result['exact_count_accuracy']:>16.2%}"
            f"{result['count_mae']:>14.2f}"
            f"{result['predicted_objects']:>16}"
        )



def evaluate_breakdown(model_name, prediction_path, ground_truth):
    predictions = load_predictions(prediction_path)
    labels = ["person", "vehicle", "cat", "dog"]

    stats = {
        label: {
            "ground_truth": 0,
            "predicted": 0,
            "correct_location": 0,
            "matched_ious": [],
            "absolute_count_error": 0,
        }
        for label in labels
    }

    category_stats = {
        category: {
            "images": 0,
            "exact_count_images": 0,
            "ground_truth": 0,
            "predicted": 0,
            "count_error": 0,
        }
        for category in CATEGORIES
    }

    for image_name, gt_data in ground_truth.items():
        gt_objects = gt_data["objects"]
        category = gt_data["category"]
        predicted_objects = predictions.get(
            image_name, {"predictions": []}
        ).get("predictions", [])

        category_stats[category]["images"] += 1
        category_stats[category]["ground_truth"] += len(gt_objects)
        category_stats[category]["predicted"] += len(predicted_objects)

        gt_counts = count_by_label(gt_objects)
        pred_counts = count_by_label(predicted_objects)
        image_exact = True

        for label in labels:
            gt_count = gt_counts.get(label, 0)
            pred_count = pred_counts.get(label, 0)
            difference = abs(gt_count - pred_count)

            stats[label]["ground_truth"] += gt_count
            stats[label]["predicted"] += pred_count
            stats[label]["absolute_count_error"] += difference
            category_stats[category]["count_error"] += difference

            if difference != 0:
                image_exact = False

        if image_exact:
            category_stats[category]["exact_count_images"] += 1

        for label in labels:
            gt_for_label = [o for o in gt_objects if o["label"] == label]
            pred_for_label = [o for o in predicted_objects if o["label"] == label]
            match_result = match_boxes(gt_for_label, pred_for_label)

            stats[label]["correct_location"] += match_result["correct_matches"]
            stats[label]["matched_ious"].extend(match_result["matched_ious"])

    class_results = {}

    for label, values in stats.items():
        gt_total = values["ground_truth"]
        ious = values["matched_ious"]

        class_results[label] = {
            "ground_truth": gt_total,
            "predicted": values["predicted"],
            "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0,
            "location_accuracy": round(
                values["correct_location"] / gt_total, 4
            ) if gt_total else 0,
            "count_mae_per_image": round(
                values["absolute_count_error"] / len(ground_truth), 4
            ),
        }

    category_results = {}

    for category, values in category_stats.items():
        images = values["images"]
        category_results[category] = {
            "images": images,
            "ground_truth": values["ground_truth"],
            "predicted": values["predicted"],
            "exact_count_accuracy": round(
                values["exact_count_images"] / images, 4
            ) if images else 0,
            "count_mae": round(
                values["count_error"] / images, 4
            ) if images else 0,
        }

    return {
        "model": model_name,
        "classes": class_results,
        "categories": category_results,
    }


def print_breakdown(breakdown_results):
    print("\nDETAILED BREAKDOWN")
    print("=" * 80)

    for model in breakdown_results:
        print(f"\n{model['model'].upper()}")
        print("\nBy class:")

        for label, values in model["classes"].items():
            print(
                f"  {label:<8} "
                f"GT={values['ground_truth']:<4} "
                f"Pred={values['predicted']:<4} "
                f"IoU={values['mean_iou']:.3f} "
                f"Loc={values['location_accuracy']:.2%} "
                f"CountMAE={values['count_mae_per_image']:.2f}"
            )

        print("\nBy dataset category:")

        for category, values in model["categories"].items():
            print(
                f"  {category:<9} "
                f"Images={values['images']:<3} "
                f"GT={values['ground_truth']:<4} "
                f"Pred={values['predicted']:<4} "
                f"CountAcc={values['exact_count_accuracy']:.2%} "
                f"MAE={values['count_mae']:.2f}"
            )

def main():
    ground_truth = load_ground_truth()

    print(
        f"Loaded ground truth for "
        f"{len(ground_truth)} images."
    )

    results = []
    breakdown_results = []

    for model_name, path in MODELS.items():
        if not path.exists():
            print(
                f"\\nSkipping {model_name}: "
                f"predictions file not found."
            )
            continue

        print(f"\\nEvaluating {model_name}...")

        result = evaluate_model(
            model_name,
            path,
            ground_truth,
        )
        results.append(result)

        breakdown_results.append(
            evaluate_breakdown(
                model_name,
                path,
                ground_truth,
            )
        )

    print_summary(results)
    print_breakdown(breakdown_results)

    save_json(results)
    save_csv(results)

    breakdown_path = RESULTS_DIR / "benchmark_breakdown.json"

    with breakdown_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            breakdown_results,
            file,
            indent=2,
        )

    print()
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved CSV:  {OUTPUT_CSV}")
    print(f"Saved breakdown: {breakdown_path}")


if __name__ == "__main__":
    main()