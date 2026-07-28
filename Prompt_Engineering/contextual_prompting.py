'''
6. Contextual Prompting

Concept: Provide background information before extraction.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Contextual Prompting")
st.markdown("Concept: Provide background information before extraction.")


user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
                    {
                "role": "user",
                "content": f"""
                            This invoice data will be stored in an ERP system.
                            Date format : YYYY-MON-DD

                            Extract:
                            - Invoice Number
                            - Vendor Name
                            - Invoice Date
                            - Payment Due Date
                            - Total Amount

                            Invoice:
                            {user_input}
                            """
                                        }
            
                ])
    
    st.write(response.choices[0].message.content)