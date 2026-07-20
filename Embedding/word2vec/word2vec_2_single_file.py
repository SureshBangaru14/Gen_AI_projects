from gensim.models import Word2Vec
from pypdf import PdfReader
import warnings
import re
import os
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

warnings.filterwarnings("ignore")

############ Step-1 : Read PDF FILE

pdf_path = r"/home/suresh/Gen_AI_Practice/Generative AI/data/resume-sample.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page_num, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""

    # print(f"\n--- Page {page_num} ---")
    # print(page_text)

    full_text += page_text + "\n"

print("\nTotal characters:", len(full_text))

############ Step-2 : Convert Text into Sentence

sentences = nltk.sent_tokenize(full_text)

# print(sentences)


############ Step-3 : Convert Sentence  into word


word_sentences = [nltk.word_tokenize(re.sub(r'[^a-zA-Z0-9\s]', '', sentence.lower()))
                  for sentence in sentences]

print(word_sentences)


############ Step-4 : Word2Vec Training

model = Word2Vec(
    sentences=word_sentences,
    vector_size=50,
    window=5,
    min_count=1,
    sg=1
)

print(model)

# model.save("resume_sample_word2vec.model")

print(model.wv["center"])


############ Step-5 : Cosine Similarity

print(model.wv.most_similar("background"))