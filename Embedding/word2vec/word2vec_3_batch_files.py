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

pdf_folder = r"/home/suresh/Gen_AI_Practice/Generative AI/data/resumes"

full_text = ""

# get all files from folder
for file_name in os.listdir(pdf_folder):

    # only process .pdf files
    if file_name.lower().endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, file_name)

        print("Processing:", file_name)

        reader = PdfReader(pdf_path)

        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"


print("Total characters:", len(full_text))

print(full_text)

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

# model.save("batch_resume_samples_word2vec.model")

print(model.wv["data"])


############ Step-5 : Cosine Similarity

print(model.wv.most_similar("data"))