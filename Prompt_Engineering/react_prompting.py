'''
10. ReAct Prompting

Concept: Reason + use external actions/tools when needed.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("ReAct Prompting")
st.markdown("Concept: Reason + use external actions/tools when needed.")


user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
                    {
                    "role": "user",
                    "content": f"""
                                Extract invoice details.

                                If information is missing:
                                - Check available document context.
                                - Validate the extracted values.
                                - Then return final JSON.

                                Invoice:
                                {user_input}
                                """
                                        }
            
                ])
    
    st.write(response.choices[0].message.content)