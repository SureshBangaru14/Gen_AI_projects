'''
2. One-shot Prompting

Concept: Give one example before asking the actual question.

'''

import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st


load_dotenv("/home/suresh/Gen_AI_Practice/.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


st.title("One-shot Prompting")
st.markdown("Concept: Give one example before asking the actual question. for the reference purpose.")

user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages= [{
            
                    "role": "user",
                    "content": f"""
                    Example:
                            Invoice:
                            Invoice No: INV-001
                            Customer: John
                            Total: $250

                            Output:
                            {{
                            "invoice_number":"INV-001",
                            "customer":"John",
                            "total":"$250"
                            }}

                            Now extract data from this invoice:

                            Invoice:
                            {user_input}
                                        """
                                        
                    }])
    
    st.write(response.choices[0].message.content)