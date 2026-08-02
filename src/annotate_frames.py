import json
import cv2
from pathlib import Path

SELECTED_FRAMES_DIR = Path("selected_frames")
RESULTS_FILE = Path("output/results.json")
ANNOTATED_DIR = Path("annotated_frames")


def draw_detection(image, detection):
    label = detection.get("label", "object")
    box = detection.get("box_2d")

    if not box or len(box) != 4:
        return

    ymin, xmin, ymax, xmax = box

    height, width = image.shape[:2]


    xmin = int(xmin / 1000 * width)
    xmax = int(xmax / 1000 * width)
    ymin = int(ymin / 1000 * height)
    ymax = int(ymax / 1000 * height)

    xmin = max(0, min(xmin, width - 1))
    xmax = max(0, min(xmax, width - 1))
    ymin = max(0, min(ymin, height - 1))
    ymax = max(0, min(ymax, height - 1))

    cv2.rectangle(
        image,
        (xmin, ymin),
        (xmax, ymax),
        (0, 255, 0),
        3
    )

    label_y = max(ymin - 10, 30)

    cv2.putText(
        image,
        label,
        (xmin, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )


def annotate_frame(image_path, result, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    detections = result.get("detections", [])

    for detection in detections:
        draw_detection(image, detection)

    cv2.imwrite(str(output_path), image)


def annotate_frames(results_file, selected_frames_dir, annotated_dir):
    annotated_dir.mkdir(exist_ok=True)

    with open(results_file, "r") as file:
        results = json.load(file)

    for result in results:
        frame_name = result.get("frame")

        if not frame_name:
            continue

        if "error" in result:
            print(f"Skipping {frame_name} because it has an error.")
            continue

        image_path = selected_frames_dir / frame_name
        output_path = annotated_dir / frame_name

        print(f"Annotating {frame_name}...")
        annotate_frame(image_path, result, output_path)

    print(f"Annotated frames saved to {annotated_dir}")


if __name__ == "__main__":
    annotate_frames(RESULTS_FILE, SELECTED_FRAMES_DIR, ANNOTATED_DIR)