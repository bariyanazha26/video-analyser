import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
IMAGE_PATH = Path("selected_frames/frame_0001.jpg")  

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

client = genai.Client(api_key=API_KEY)

prompt = """
Analyse this image for an AI-powered security system.

Return only valid JSON with:
{
  "people_count": number,
  "objects": [],
  "description": "",
  "security_relevance": ""
}
"""

image_bytes = IMAGE_PATH.read_bytes()

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

print(response.text)
