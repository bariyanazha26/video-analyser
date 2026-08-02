from pathlib import Path
import csv
import json


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_JSON = RESULTS_DIR / "video_model_comparison.json"
OUTPUT_CSV = RESULTS_DIR / "video_model_comparison.csv"


def find_result_files():
    return sorted(
        RESULTS_DIR.glob("*/*/*/results.json")
    )


def load_results():
    rows = []

    for path in find_result_files():
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        rows.append({
            "model": data["model"],
            "category": data["category"],
            "video": data["video"],
            "resolution": data["image_size"],
            "frames_processed": data["total_frames"],
            "total_detections": data[
                "total_person_detections"
            ],
            "average_detections_per_frame": data[
                "average_detections_per_frame"
            ],
            "average_inference_seconds": data[
                "average_inference_seconds"
            ],
            "processing_fps": data["processing_fps"],
        })

    return rows


def save_json(rows):
    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rows,
            file,
            indent=2,
        )


def save_csv(rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

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
        writer.writerows(rows)


def print_table(rows):
    print("\nVIDEO MODEL COMPARISON")
    print("=" * 120)

    print(
        f"{'Model':<12}"
        f"{'Category':<12}"
        f"{'Resolution':>12}"
        f"{'Frames':>10}"
        f"{'Total Det.':>12}"
        f"{'Avg Det./Frame':>16}"
        f"{'Avg Time':>12}"
        f"{'FPS':>10}"
    )

    print("-" * 120)

    for row in rows:
        print(
            f"{row['model']:<12}"
            f"{row['category']:<12}"
            f"{row['resolution']:>12}"
            f"{row['frames_processed']:>10}"
            f"{row['total_detections']:>12}"
            f"{row['average_detections_per_frame']:>16.2f}"
            f"{row['average_inference_seconds']:>11.3f}s"
            f"{row['processing_fps']:>10.2f}"
        )


def main():
    rows = load_results()

    if not rows:
        raise FileNotFoundError(
            f"No benchmark results found in: {RESULTS_DIR}"
        )

    rows.sort(
        key=lambda item: (
            item["model"],
            item["category"],
            item["resolution"],
        )
    )

    print_table(rows)
    save_json(rows)
    save_csv(rows)

    print("\nSaved files:")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()