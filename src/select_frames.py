from pathlib import Path
import shutil


def select_frames(frames_dir, selected_dir, sample_rate):
    selected_dir.mkdir(exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_files:
        print("No frames found.")
        return []

    selected_files = []

    for index, frame_file in enumerate(frame_files):
        if index % sample_rate == 0:
            new_name = selected_dir / frame_file.name
            shutil.copy(frame_file, new_name)
            selected_files.append(new_name)

    print(f"Selected {len(selected_files)} frames.")
    return selected_files


if __name__ == "__main__":
    frames_dir = Path("frames")
    selected_dir = Path("selected_frames")
    sample_rate = 30

    select_frames(frames_dir, selected_dir, sample_rate)