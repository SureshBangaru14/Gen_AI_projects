'''
Few-shot Prompting

Concept: Give multiple examples before asking.

'''
import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Few-shot Prompting")
st.markdown("Concept: Give multiple examples before asking.")

user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[{
            
                    "role" : "user",
                    "content" : f"""
                    
                    Example 1:

                        Invoice:
                        Invoice No: INV-001
                        Customer: John
                        Total: $250

                        Output:
                        {{"invoice_number":"INV-001","customer":"John","total":"$250"}}


                    Example 2:

                        Invoice:
                        Invoice No: INV-002
                        Customer: Alice
                        Total: $500

                        Output:
                        {{"invoice_number":"INV-002","customer":"Alice","total":"$500"}}
                        
                    Now extract :
                    
                    Invoice :
                    {user_input}
                    
                    """
                    }])
    
    st.write(response.choices[0].message.content)