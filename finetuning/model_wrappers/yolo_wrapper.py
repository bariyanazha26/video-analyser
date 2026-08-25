from ultralytics import YOLO


MODEL_NAME = "yolov8m.pt"
CONFIDENCE = 0.30

model = YOLO(MODEL_NAME)


def detect(image_path):
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        verbose=False,
    )

    predictions = []

    result = results[0]

    for box in result.boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])

        original_label = (
            model.names[class_id]
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

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        predictions.append(
            {
                "label": label,
                "score": round(score, 4),
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
            }
        )

    return predictions