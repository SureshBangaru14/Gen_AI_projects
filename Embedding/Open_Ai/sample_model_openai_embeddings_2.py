import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Load .env
load_dotenv("/home/suresh/Gen_AI_Practice/.env")

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

texts = [
    "I like data science and machine learning.",
    "Artificial Intelligence is transforming industries.",
    "Python is a popular programming language.",
    "OpenAI provides powerful AI models."
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

# Get embeddings
embeddings = [item.embedding for item in response.data]

for i, emb in enumerate(embeddings):
    print(f"\nText {i+1}: {texts[i]}")
    print("Vector size:", len(emb))
    print("First 10 values:", emb[:10])