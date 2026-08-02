import cv2
from pathlib import Path


def extract_frames(video_path, output_folder):
    output_folder.mkdir(exist_ok=True)

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_number = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        frame_number += 1
        frame_name = output_folder / f"frame_{frame_number:04d}.jpg"
        cv2.imwrite(str(frame_name), frame)

    video.release()

    print(f"Done. Saved {frame_number} frames.")


if __name__ == "__main__":
    video_path = Path("input/sample.mp4")
    output_folder = Path("frames")

    extract_frames(video_path, output_folder)