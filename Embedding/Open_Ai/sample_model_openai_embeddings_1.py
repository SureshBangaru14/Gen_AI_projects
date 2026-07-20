import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Load .env
load_dotenv("/home/suresh/Gen_AI_Practice/.env")

api_key_1 = os.getenv("OPENAI_API_KEY")

print("API Key Loaded:", api_key_1 is not None)
# Avoid printing the full API key in real applications.

client = OpenAI(api_key=api_key_1)

text = "I like data science and machine learning."

response = client.embeddings.create(
    model="text-embedding-3-small",   # text-embedding-3-large, text-embedding-ada-002
    input=text
)

embedding = response.data[0].embedding

print("Vector size:", len(embedding))
print("First 10 values:", embedding[:10])