from dotenv import load_dotenv
import os

load_dotenv(override=True)
key = os.getenv("GOOGLE_API_KEY")
if key:
    print(f"Google Key loaded. Ends with: ...{key[-4:]}")
else:
    print("No Google key found.")
