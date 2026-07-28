'''
5. Constraint Prompting

Concept: Give rules that the AI must follow.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Constraint Prompting")
st.markdown("Concept: Give rules that the AI must follow.")


user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
                    {
                        "role": "user",
                        "content": f"""
                                        Extract invoice details.

                                        Rules:
                                        1. Return only JSON.
                                        2. Do not add explanations.
                                        3. Use null for missing values.

                                        Invoice:
                                        {user_input}
                                        """}
            
                ])
    
    st.write(response.choices[0].message.content)