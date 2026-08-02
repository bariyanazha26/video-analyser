import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)

prompt = """
Analyse this image for an AI-powered security system.

Return ONLY raw valid JSON.
Do not include markdown.
Do not include ```json.
Do not include explanations.

Detect visible people, vehicles, and relevant security objects.

Use this exact structure:
{
  "people_count": 0,
  "detections": [
    {
      "label": "person",
      "box_2d": [ymin, xmin, ymax, xmax]
    }
  ],
  "description": "",
  "security_relevance": ""
}

Important:
- Return bounding boxes as normalized coordinates from 0 to 1000.
- Use [ymin, xmin, ymax, xmax] format.
- ymin and ymax represent vertical positions.
- xmin and xmax represent horizontal positions.
- Only include objects that are clearly visible.
- Make each bounding box fit the detected object as closely as possible.
- If no objects are detected, return an empty detections array.
"""

def clean_json_response(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text

def analyse_frame(image_path):
    image_bytes = image_path.read_bytes()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_bytes,
                }
            },
        ],
    )

    return response.text

def analyse_selected_frames(selected_frames_dir, output_file):
    output_file.parent.mkdir(exist_ok=True)

    image_files = sorted(selected_frames_dir.glob("*.jpg"))

    if not image_files:
        print("No selected frames found.")
        return []

    results = []

    for image_file in image_files:
        print(f"Analysing {image_file.name}...")

        try:
            result_text = analyse_frame(image_file)
            cleaned_text = clean_json_response(result_text)
            result_json = json.loads(cleaned_text)

            result_json["frame"] = image_file.name
            results.append(result_json)

        except Exception as error:
            print(f"Error analysing {image_file.name}: {error}")
            results.append({
                "frame": image_file.name,
                "error": str(error)
            })

        time.sleep(5)

    with open(output_file, "w") as file:
        json.dump(results, file, indent=2)

    print(f"Saved results to {output_file}")
    return results

if __name__ == "__main__":
    analyse_selected_frames(
        Path("selected_frames"),
        Path("output/results.json")
    )