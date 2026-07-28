'''
4. Role Prompting

Concept: Give the AI a specific role.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Role Prompting")
st.markdown("Concept: Give the AI a specific role.")


user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
            
                    {
                        "role": "system",
                        "content": "You are an expert Accounts Payable Specialist."
                    },
                    {
                        "role": "user",
                        "content": f"""Extract invoice details and return structured JSON.

                            
                            Invoice :
                            {user_input}"""}
            
                ])
    
    st.write(response.choices[0].message.content)