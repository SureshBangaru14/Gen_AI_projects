'''
8. Instruction Prompting

Concept: Give clear step-by-step instructions.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Instruction Prompting")
st.markdown("Concept: Give clear step-by-step instructions.")


user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
                    {
                    "role": "user",
                    "content": f"""
                            Instructions:

                            1. Read the invoice.
                            2. Identify invoice number.
                            3. Identify customer name.
                            4. Identify total amount.
                            5. Return JSON format.

                            Invoice:
                            {user_input}
                            """
                                    }
            
                ])
    
    st.write(response.choices[0].message.content)