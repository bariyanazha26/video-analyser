# Video Analyser Prototype

## Overview

This project was developed as part of my internship at Optara Systems.

The system analyses CCTV footage by extracting video frames, selecting representative frames, detecting people, analysing scenes using Google Gemini, and producing structured JSON outputs. The project also includes benchmarking pipelines for comparing multiple object detection models on both images and videos.

---

## Features

### Video Analysis Pipeline
- Extract frames from video
- Sample representative frames
- Detect people in frames
- Analyse frames using Google Gemini
- Generate structured JSON results
- Annotate frames with detections
- Run the complete pipeline from a single command

### Image Benchmarking
- Evaluate multiple object detection models
- Compare detections against LabelMe ground truth
- Calculate:
  - Precision
  - Recall
  - F1 Score
  - True Positives
  - False Positives
  - False Negatives
- Generate comparison tables and JSON summaries

### Video Benchmarking
- Benchmark multiple YOLO models on CCTV videos
- Evaluate sparse, normal and crowded scenes
- Compare 640 × 640 and 1280 × 1280 resolutions
- Measure:
  - Processing FPS
  - Inference time
  - Average people detected per frame
- Export annotated videos
- Export JSON and CSV summaries
- Run all benchmarks from a single command

---

## Project Structure

```
video-analyser/
│
├── src/                     # Main video analysis pipeline
├── input/                   # Input videos
├── output/                  # Pipeline outputs
│
├── evaluation/              # Image benchmarking
├── evaluation_dataset/      # Labelled evaluation dataset
│
├── video_benchmark/         # Video benchmarking
│
├── YOLOX/                   # YOLOX implementation
│
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- OpenCV
- Ultralytics
- Google GenAI SDK
- python-dotenv
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Video Analysis Pipeline

Place a video inside:

```
input/
```

Run:

```bash
python src/main.py
```

Outputs include:

- Selected frames
- Annotated frames
- JSON analysis

---

## Running Image Benchmarking

```bash
python evaluation/run_yolov8.py
```

Evaluate all models:

```bash
python evaluation/evaluate_models.py
```

Outputs:

- Detection results
- Precision / Recall / F1
- Model comparison

---

## Running Video Benchmarking

Run all benchmarks:

```bash
python video_benchmark/run_video_benchmark.py
```

Generate the comparison table:

```bash
python video_benchmark/compare_video_models.py
```

Outputs:

- Annotated benchmark videos
- JSON summaries
- CSV comparison table

---

## Technologies

- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- YOLOX
- Google Gemini API
- LabelMe
- Git
- GitHub

---

## Future Improvements

- Real-time CCTV processing
- Live webcam support
- Additional object detection models
- Cloud deployment
- Interactive visual dashboard