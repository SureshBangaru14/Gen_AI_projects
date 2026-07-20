"""all-MiniLM-L6-v2 is a sentence embedding model, not a text generation model like GPT. 
Its job is to convert text (sentences or short paragraphs) into numerical vectors (embeddings) that capture the semantic meaning of the text.

Breaking down the name
all : Trained on many different datasets, making it useful for general-purpose text understanding.
MiniLM : A lightweight Transformer architecture that is much smaller and faster than models like BERT while retaining good performance.
L6 : The model has 6 Transformer layers ("L6" = Layer 6).
v2 : The second improved version of the model.
    """



from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


documents = [
    "Python developer with AWS experience",
    "Java backend engineer",
    "Machine learning engineer"
]

######### Custom data Embedding
doc_embeddings = model.encode(documents)


query = "AWS Python developer"

######### Query data Embedding
query_embedding = model.encode([query])


######### Finding Similarity between Custom data and Query data
similarity = cosine_similarity(
    query_embedding,
    doc_embeddings
)


print((similarity))


df = pd.DataFrame(
    similarity.flatten(),
    columns=["match_probability"]
)
df["data"] = documents
# print(df)

######### Finding Similarity Greater than 0.5 and top3 display
result =  df[df["match_probability"] > 0.5].nlargest(3, "match_probability")
print(result)
