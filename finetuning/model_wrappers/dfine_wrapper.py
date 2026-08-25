from PIL import Image
import torch

from transformers import (
    AutoImageProcessor,
    DFineForObjectDetection,
)


MODEL_NAME = "ustc-community/dfine_x_coco"
CONFIDENCE = 0.30


processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = DFineForObjectDetection.from_pretrained(
    MODEL_NAME
)

model.eval()


def detect(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    results = (
        processor.post_process_object_detection(
            outputs,
            target_sizes=[
                (
                    image.height,
                    image.width,
                )
            ],
            threshold=CONFIDENCE,
        )[0]
    )

    predictions = []

    for score, label_id, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"],
    ):
        original_label = (
            model.config.id2label[
                int(label_id)
            ]
            .strip()
            .lower()
        )

        if original_label == "person":
            label = "person"

        elif original_label in {
            "car",
            "truck",
            "bus",
            "motorcycle",
        }:
            label = "vehicle"

        elif original_label == "cat":
            label = "cat"

        elif original_label == "dog":
            label = "dog"

        else:
            continue

        x1, y1, x2, y2 = box.tolist()

        predictions.append(
            {
                "label": label,
                "score": round(
                    float(score),
                    4,
                ),
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
            }
        )

    return predictions