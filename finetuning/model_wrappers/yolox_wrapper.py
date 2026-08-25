from pathlib import Path
import sys

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

YOLOX_DIR = PROJECT_DIR.parent / "YOLOX"

print(f"YOLOX directory: {YOLOX_DIR}")
print(f"YOLOX exists: {YOLOX_DIR.exists()}")

# Add cloned YOLOX repo to Python's import path
sys.path.insert(0, str(YOLOX_DIR))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import cv2
import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CHECKPOINT_PATH = YOLOX_DIR / "yolox_m.pth"

CONFIDENCE = 0.30
NMS_THRESHOLD = 0.45
TEST_SIZE = (640, 640)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

exp = get_exp(
    str(
        YOLOX_DIR
        / "exps"
        / "default"
        / "yolox_m.py"
    ),
    None,
)

model = exp.get_model()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


# --------------------------------------------------
# DETECTION
# --------------------------------------------------

def detect(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    processed, ratio = preproc(
        image,
        TEST_SIZE,
    )

    tensor = (
        torch.from_numpy(processed)
        .unsqueeze(0)
        .float()
    )

    with torch.no_grad():
        outputs = model(tensor)

        outputs = postprocess(
            outputs,
            num_classes=80,
            conf_thre=CONFIDENCE,
            nms_thre=NMS_THRESHOLD,
            class_agnostic=True,
        )

    predictions = []

    detections = outputs[0]

    if detections is None:
        return predictions

    detections = detections.cpu()

    boxes = detections[:, 0:4] / ratio

    scores = (
        detections[:, 4]
        * detections[:, 5]
    )

    class_ids = detections[:, 6]

    for box, score, class_id in zip(
        boxes,
        scores,
        class_ids,
    ):
        original_label = (
            COCO_CLASSES[int(class_id)]
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