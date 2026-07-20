import os
from dotenv import load_dotenv
import voyageai

# Load .env
load_dotenv("/home/suresh/Gen_AI_Practice/.env")

# Read API key
api_key = os.getenv("VOYAGE_API_KEY")

# Create client
client = voyageai.Client(api_key=api_key)

texts = [
    "I like data science and machine learning.",
    "Artificial Intelligence is transforming industries.",
    "Python is a popular programming language.",
    "OpenAI provides powerful AI models."
]

response = client.embed(
    texts=texts,
    model="voyage-3-lite"
)

embeddings = response.embeddings

for i, emb in enumerate(embeddings):
    print(f"\nText {i+1}: {texts[i]}")
    print("Vector size:", len(emb))
    print("First 10 values:", emb[:10])