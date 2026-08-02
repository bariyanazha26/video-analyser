from pathlib import Path
import json
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "evaluation_dataset"
RESULTS_DIR = BASE_DIR / "results"

MODELS = [
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolox_x",
    "yolo11m",
    "rtdetr_l",
    "faster_rcnn",
    "retinanet",
    "grounding_dino",
    "deformable_detr",
]

CATEGORIES = [
    "crowded",
    "normal",
    "sparse",
    "difficult_negative",
]

IOU_THRESHOLD = 0.5


def normalise_box(x1, y1, x2, y2):
    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    ]


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(image_path):
    json_path = image_path.with_suffix(".json")

    # Missing JSON = intentionally no people
    if not json_path.exists():
        return []

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    boxes = []

    for shape in data.get("shapes", []):
        if shape.get("label") != "person":
            continue

        if shape.get("shape_type") != "rectangle":
            continue

        points = shape.get("points", [])

        if len(points) != 2:
            continue

        (x1, y1), (x2, y2) = points

        boxes.append(
            normalise_box(x1, y1, x2, y2)
        )

    return boxes


def match_predictions(gt_boxes, predicted_boxes):
    matched_gt = set()

    tp = 0
    fp = 0

    # Highest confidence predictions first
    predicted_boxes = sorted(
        predicted_boxes,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    for prediction in predicted_boxes:
        pred_box = prediction["box"]

        best_iou = 0
        best_gt_index = None

        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= IOU_THRESHOLD
        ):
            tp += 1
            matched_gt.add(best_gt_index)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


def calculate_metrics(tp, fp, fn):
    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return precision, recall, f1


def load_model_results(model_name):
    path = RESULTS_DIR / model_name / "results.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find results for {model_name}: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    return {
        (item["category"], item["image"]): item
        for item in results
    }


def evaluate_model(model_name):
    predictions = load_model_results(model_name)

    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "time": 0,
        "images": 0,
    }

    category_totals = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "time": 0,
            "images": 0,
        }
    )

    for category in CATEGORIES:
        category_dir = DATASET_DIR / category

        for image_path in sorted(category_dir.glob("*.jpg")):
            gt_boxes = load_ground_truth(image_path)

            key = (category, image_path.name)

            if key not in predictions:
                print(
                    f"WARNING: no prediction for "
                    f"{model_name}: {category}/{image_path.name}"
                )
                continue

            result = predictions[key]

            predicted_boxes = []

            for detection in result.get("detections", []):
                box = detection["box"]

                predicted_boxes.append({
                    "confidence": detection["confidence"],
                    "box": normalise_box(
                        box["x1"],
                        box["y1"],
                        box["x2"],
                        box["y2"],
                    ),
                })

            tp, fp, fn = match_predictions(
                gt_boxes,
                predicted_boxes,
            )

            inference_time = result.get(
                "inference_seconds",
                0,
            )

            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            totals["time"] += inference_time
            totals["images"] += 1

            category_totals[category]["tp"] += tp
            category_totals[category]["fp"] += fp
            category_totals[category]["fn"] += fn
            category_totals[category]["time"] += inference_time
            category_totals[category]["images"] += 1

    precision, recall, f1 = calculate_metrics(
        totals["tp"],
        totals["fp"],
        totals["fn"],
    )

    avg_time = (
        totals["time"] / totals["images"]
        if totals["images"] > 0
        else 0
    )

    return {
        "model": model_name,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_time": avg_time,
        "categories": category_totals,
    }


def print_comparison(results):
    print("\nFINAL MODEL COMPARISON")
    print("=" * 100)

    print(
        f"{'Model':<12}"
        f"{'TP':>6}"
        f"{'FP':>6}"
        f"{'FN':>6}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Avg Time':>12}"
    )

    print("-" * 100)

    for result in results:
        print(
            f"{result['model']:<12}"
            f"{result['tp']:>6}"
            f"{result['fp']:>6}"
            f"{result['fn']:>6}"
            f"{result['precision']:>12.3f}"
            f"{result['recall']:>10.3f}"
            f"{result['f1']:>10.3f}"
            f"{result['avg_time']:>11.3f}s"
        )


def print_category_breakdown(results):
    for result in results:
        print(
            f"\nCATEGORY BREAKDOWN: {result['model']}"
        )
        print("-" * 75)

        print(
            f"{'Category':<22}"
            f"{'TP':>5}"
            f"{'FP':>5}"
            f"{'FN':>5}"
            f"{'Precision':>11}"
            f"{'Recall':>9}"
            f"{'F1':>9}"
        )

        for category in CATEGORIES:
            values = result["categories"][category]

            precision, recall, f1 = calculate_metrics(
                values["tp"],
                values["fp"],
                values["fn"],
            )

            print(
                f"{category:<22}"
                f"{values['tp']:>5}"
                f"{values['fp']:>5}"
                f"{values['fn']:>5}"
                f"{precision:>11.3f}"
                f"{recall:>9.3f}"
                f"{f1:>9.3f}"
            )


def save_summary(results):
    output_path = RESULTS_DIR / "model_comparison.json"

    serialisable = []

    for result in results:
        serialisable.append({
            "model": result["model"],
            "true_positives": result["tp"],
            "false_positives": result["fp"],
            "false_negatives": result["fn"],
            "precision": round(
                result["precision"],
                4,
            ),
            "recall": round(
                result["recall"],
                4,
            ),
            "f1": round(
                result["f1"],
                4,
            ),
            "average_inference_seconds": round(
                result["avg_time"],
                4,
            ),
        })

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serialisable,
            file,
            indent=2,
        )

    print(
        f"\nSaved comparison to: {output_path}"
    )


def main():
    results = []

    for model_name in MODELS:
        print(f"Evaluating {model_name}...")
        results.append(
            evaluate_model(model_name)
        )

    # Best F1 first
    results.sort(
        key=lambda item: item["f1"],
        reverse=True,
    )

    print_comparison(results)
    print_category_breakdown(results)
    save_summary(results)


if __name__ == "__main__":
    main()