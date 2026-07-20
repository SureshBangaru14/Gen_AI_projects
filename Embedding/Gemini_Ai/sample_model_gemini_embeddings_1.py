import os
from dotenv import load_dotenv
from google import genai

# Load .env
load_dotenv("/home/suresh/Gen_AI_Practice/.env")

# Read API key
api_key = os.getenv("GEMINI_API_KEY")

# Create client
client = genai.Client(api_key=api_key)

texts = [
    "I like data science and machine learning.",
    "Artificial Intelligence is transforming industries.",
    "Python is a popular programming language.",
    "OpenAI provides powerful AI models."
]

embeddings = []

for text in texts:
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )

    embeddings.append(response.embeddings[0].values)

for i, emb in enumerate(embeddings):
    print(f"\nText {i+1}: {texts[i]}")
    print("Vector size:", len(emb))
    print("First 10 values:", emb[:10])