from pathlib import Path
from extract_frames import extract_frames
from select_frames import select_frames
from analyse_selected_frames import analyse_selected_frames
from annotate_frames import annotate_frames

VIDEO_PATH = Path("input/sample.mp4")
FRAMES_DIR = Path("frames")
SELECTED_FRAMES_DIR = Path("selected_frames")
OUTPUT_FILE = Path("output/results.json")
ANNOTATED_DIR = Path("annotated_frames")
SAMPLE_RATE = 30

def main():
    print("Starting video analyser pipeline...")

    print("Step 1: Extracting frames...")
    extract_frames(VIDEO_PATH, FRAMES_DIR)

    print("Step 2: Selecting frames...")
    select_frames(FRAMES_DIR, SELECTED_FRAMES_DIR, SAMPLE_RATE)

    print("Step 3: Analysing selected frames...")
    analyse_selected_frames(SELECTED_FRAMES_DIR, OUTPUT_FILE)
    
    print("Step 4: Annotating frames...")
    annotate_frames(OUTPUT_FILE, SELECTED_FRAMES_DIR, ANNOTATED_DIR)

    print("Pipeline finished.")

if __name__ == "__main__":
    main()