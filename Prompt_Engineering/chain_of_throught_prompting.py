'''
9. Chain-of-Thought Prompting

Concept: Ask the model to analyze before answering it means give instructions logic step by step.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Chain-of-Thought Prompting")
st.markdown("Concept: Ask the model to analyze before answering it means give instructions logic step by step.")


user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
                    {
                    "role": "user",
                    "content": f"""
                                Analyze the invoice carefully.

                                Identify each field step by step.

                                Then provide the final JSON output.

                                Invoice:
                                {user_input}
                                """
                                        }
            
                ])
    
    st.write(response.choices[0].message.content)