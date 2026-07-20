import os
from dotenv import load_dotenv
import cohere

# Load .env
load_dotenv("/home/suresh/Gen_AI_Practice/.env")

# Read API key
api_key = os.getenv("COHERE_API_KEY")

# Create client
Client = cohere.Client(api_key)

texts = [
    "I like data science and machine learning.",
    "Artificial Intelligence is transforming industries.",
    "Python is a popular programming language.",
    "OpenAI provides powerful AI models."
]

response = Client.embed(
    texts=texts,
    model="embed-english-v3.0",
    input_type="search_document"
)

embeddings = response.embeddings

for i, emb in enumerate(embeddings):
    print(f"\nText {i+1}: {texts[i]}")
    print("Vector size:", len(emb))
    print("First 10 values:", emb[:10])