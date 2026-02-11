import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("No GOOGLE_API_KEY found.")
    exit(1)

genai.configure(api_key=api_key)

print("Listing models...")
try:
    with open("models_utf8.txt", "w", encoding="utf-8") as f:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
                f.write(m.name + "\n")
except Exception as e:
    print(f"Error listing models: {e}")
