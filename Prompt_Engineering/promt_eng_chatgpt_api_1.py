import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv("/home/suresh/Gen_AI_Practice/.env")

api_key_1 = os.getenv("OPENAI_API_KEY")

print("API Key Loaded:", api_key_1 is not None)
# Avoid printing the full API key in real applications.

client = OpenAI(api_key=api_key_1)


while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "your help full for AI assistant"
            },
            {
                "role": "user",
                "content": user_input
            }])
    
    user_input = response.choices[0].message.content
    print(user_input)