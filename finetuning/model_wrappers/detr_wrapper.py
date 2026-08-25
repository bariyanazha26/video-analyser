from PIL import Image
import torch

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)


MODEL_NAME = "facebook/detr-resnet-50"

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME
)

model.eval()


def detect(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [[image.height, image.width]]
    )

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.30,
    )[0]

    predictions = []

    for score, label_id, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"],
    ):
        label = model.config.id2label[
            int(label_id)
        ].lower()

        # Only keep classes relevant to your dataset
        if label == "person":
            output_label = "person"

        elif label in {
            "car",
            "truck",
            "bus",
            "motorcycle",
        }:
            output_label = "vehicle"

        elif label == "cat":
            output_label = "cat"

        elif label == "dog":
            output_label = "dog"

        else:
            continue

        x1, y1, x2, y2 = box.tolist()

        predictions.append(
            {
                "label": output_label,
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