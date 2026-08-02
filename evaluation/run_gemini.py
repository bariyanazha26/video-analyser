import json
import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "evaluation_dataset"

RESULTS_DIR = BASE_DIR / "results" / "gemini"
JSON_DIR = RESULTS_DIR / "json"
ANNOTATED_DIR = RESULTS_DIR / "annotated"

MODEL_NAME = "gemini-2.5-flash"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MAX_RETRIES = 3
DELAY_BETWEEN_IMAGES = 3


class DetectedObject(BaseModel):
    label: str = Field(
        description="Name of the detected object."
    )
    box_2d: list[int] = Field(
        description=(
            "Bounding box as [ymin, xmin, ymax, xmax], "
            "normalised from 0 to 1000."
        )
    )


class GeminiAnalysis(BaseModel):
    people_count: int
    objects: list[DetectedObject]
    description: str
    security_relevance: str


def draw_label(
    image,
    text: str,
    x1: int,
    y1: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    label_top = max(
        y1 - text_height - baseline - 8,
        0,
    )
    label_bottom = label_top + text_height + baseline + 8

    cv2.rectangle(
        image,
        (x1, label_top),
        (x1 + text_width + 8, label_bottom),
        (0, 200, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x1 + 4, label_bottom - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate_image(
    image_path: Path,
    analysis: dict,
) -> None:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not annotate {image_path.name}")
        return

    height, width = image.shape[:2]

    for detected_object in analysis.get("objects", []):
        box = detected_object.get("box_2d", [])

        if len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = box

        # Gemini returns coordinates normalised from 0 to 1000.
        x1 = int((xmin / 1000) * width)
        y1 = int((ymin / 1000) * height)
        x2 = int((xmax / 1000) * width)
        y2 = int((ymax / 1000) * height)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2,
        )

        draw_label(
            image,
            detected_object.get("label", "object"),
            x1,
            y1,
        )

    output_path = ANNOTATED_DIR / image_path.name

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(
            f"Could not save annotated image: {output_path}"
        )


def analyse_image(
    client: genai.Client,
    image_path: Path,
) -> dict:
    image_bytes = image_path.read_bytes()

    suffix = image_path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_types[suffix],
    )

    prompt = """
Analyse this CCTV image for an object-detection comparison.

Detect every clearly visible object relevant to surveillance, including:
people, vehicles, bags, furniture, animals, machinery, equipment,
signs and other important objects.

For every detected object:
- provide a concise object label
- provide one bounding box
- express the box as [ymin, xmin, ymax, xmax]
- normalise all coordinates from 0 to 1000
- do not create duplicate boxes for the same object
- do not include objects that are too uncertain to identify

Also provide:
- the total number of visible people
- a concise scene description
- any security-relevant observations

Only report objects actually visible in the image.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            prompt,
            image_part,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiAnalysis,
            temperature=0,
        ),
    )

    if response.parsed is not None:
        return response.parsed.model_dump()

    # Fallback if parsed output is unavailable.
    return json.loads(response.text)


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in your .env file."
        )

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in DATASET_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {DATASET_DIR}"
        )

    client = genai.Client(api_key=api_key)

    combined_results = []

    print(f"Found {len(image_paths)} images.")
    print(f"Using model: {MODEL_NAME}")

    for index, image_path in enumerate(image_paths, start=1):
        json_path = JSON_DIR / f"{image_path.stem}.json"

        # Avoid charging the API again if an image was already completed.
        if json_path.exists():
            print(
                f"[{index}/{len(image_paths)}] "
                f"Skipping existing result: {image_path.name}"
            )

            with json_path.open("r", encoding="utf-8") as file:
                stored_result = json.load(file)

            combined_results.append(stored_result)
            continue

        print(
            f"[{index}/{len(image_paths)}] "
            f"Processing {image_path.name}"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                analysis = analyse_image(
                    client,
                    image_path,
                )

                stored_result = {
                    "frame": image_path.name,
                    "model": MODEL_NAME,
                    **analysis,
                }

                with json_path.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        stored_result,
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )

                annotate_image(
                    image_path,
                    stored_result,
                )

                combined_results.append(stored_result)

                print(
                    f"Saved Gemini result for "
                    f"{image_path.name}"
                )
                break

            except Exception as error:
                print(
                    f"Attempt {attempt}/{MAX_RETRIES} failed: "
                    f"{error}"
                )

                if attempt == MAX_RETRIES:
                    error_result = {
                        "frame": image_path.name,
                        "model": MODEL_NAME,
                        "error": str(error),
                    }

                    combined_results.append(error_result)
                    print(
                        f"Could not process {image_path.name}"
                    )
                else:
                    wait_time = 10 * attempt
                    print(
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

        time.sleep(DELAY_BETWEEN_IMAGES)

    combined_path = RESULTS_DIR / "results.json"

    with combined_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combined_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nGemini evaluation complete.")
    print(f"JSON results: {JSON_DIR}")
    print(f"Annotated images: {ANNOTATED_DIR}")
    print(f"Combined results: {combined_path}")


if __name__ == "__main__":
    main()