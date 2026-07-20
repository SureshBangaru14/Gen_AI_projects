import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv("/home/suresh/Gen_AI_Practice/.env")

api_key_1 = os.getenv("OPENAI_API_KEY")

print("API Key Loaded:", api_key_1 is not None)
# Avoid printing the full API key in real applications.

client = OpenAI(api_key=api_key_1)
