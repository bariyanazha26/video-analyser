from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

VIDEOS_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
ANNOTATED_DIR = BASE_DIR / "annotated"


VIDEOS = {
    "sparse": VIDEOS_DIR / "train_station_cctv.mp4",
    "normal": VIDEOS_DIR / "street_cctv.mp4",
    "crowded": VIDEOS_DIR / "shopping_centre_cctv.mp4",
}


RESOLUTIONS = [
    640,
    1280,
]


CONF_THRESHOLD = 0.25


MODELS = {
    "yolov8s": {
        "weights": "yolov8s.pt",
    },
    "yolov8m": {
        "weights": "yolov8m.pt",
    },
    "yolov8l": {
        "weights": "yolov8l.pt",
    },
    "yolo11m": {
        "weights": "yolo11m.pt",
    },
}