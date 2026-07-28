'''
11. Tree-of-Thought (ToT) Prompting

Concept: Explore multiple possible solutions, compare them, and select the best approach.

'''



import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv("/home/suresh/Gen_AI_Practice/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Tree-of-Thought (ToT) Prompting")
st.markdown("Concept: Explore multiple possible solutions, compare them, and select the best approach.")


user_input = st.text_area("Paste or type the Invoice text below")

if st.button("Extract"):
    response = client.chat.completions.create(
        
        model = "gpt-4.1-mini",
        messages=[
                    {
                        "role": "user",
                        "content": f"""
                                        Analyze the invoice extraction problem using multiple approaches.

                                        Approach 1:
                                        - Extract fields using invoice labels.

                                        Approach 2:
                                        - Extract fields using document structure.

                                        Approach 3:
                                        - Extract fields using semantic understanding.

                                        Compare all approaches.

                                        Select the most accurate approach and return the final JSON output.

                                        Invoice:
                                        {user_input}
                                        """
                                                }
            
                ])
    
    st.write(response.choices[0].message.content)