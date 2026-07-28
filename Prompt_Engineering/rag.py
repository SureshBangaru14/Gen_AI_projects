'''
12. Retrieval-Augmented Generation (RAG)

Concept: Retrieve relevant information from external documents/knowledge sources and use it to generate an accurate answer.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Retrieval-Augmented Generation (RAG)")
st.markdown("Concept: Retrieve relevant information from external documents/knowledge sources and use it to generate an accurate answer.")


user_input = st.text_area("Paste or type the Invoice text below")

# Sample retrieved documents (Normally this comes from Vector DB like FAISS, Chroma, Pinecone)
retrieved_documents = """
Company Invoice Processing Rules:

1. Invoice number is mandatory.
2. Vendor name must match supplier records.
3. Invoice date must be present.
4. Total amount must include tax.
5. Output should contain only required invoice fields.
"""

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
                    {
                        "role": "system",
                        "content": """
                            You are an invoice processing assistant.

                            Use the provided company invoice rules
                            and document context to extract accurate invoice information.

                            Follow the given rules while generating the response.
                            """
                                        },

                    {
                        "role": "user",
                        "content": f"""
                                Retrieved Context:

                                {retrieved_documents}


                                Task:

                                Extract invoice details according to the provided rules.

                                Required Fields:
                                - Invoice Number
                                - Vendor Name
                                - Invoice Date
                                - Customer Name
                                - Total Amount


                                Invoice:

                                {user_input}


                                Return only JSON format.
                                """}

            
                ])
    
    st.write(response.choices[0].message.content)