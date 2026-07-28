'''
7. Persona Prompting

Concept: Ask AI to behave in a specific style.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Persona Prompting")
st.markdown("Concept: Ask AI to behave in a specific style.")


user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
                    {
                    "role": "system",
                    "content": "Respond like a senior financial analyst."},
                    
                    {
                        "role": "user",
                        "content": f"""Analyze this invoice and extract important financial information and return in JSON.

                        Invoice:
                        {user_input}
                        """}
            
                ])
    
    st.write(response.choices[0].message.content)