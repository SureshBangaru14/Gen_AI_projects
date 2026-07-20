from gensim.models import Word2Vec
import warnings
import re
warnings.filterwarnings("ignore")



'''

Text Corpus
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Stopword Removal / Lemmatization
     ↓
Vocabulary Creation
     ↓
Context Window Selection
     ↓
CBOW / Skip-Gram Pair Generation
     ↓
Word2Vec Training
     ↓
Word Embeddings
     ↓
Cosine Similarity
     ↓
Similar Words

'''


#### Step 1: Text Corpus

text = "I love NLP and I love Machine Learning"


#### Step 2: Text Cleaning

# Convert to lowercase and remove punctuation.

text = text.lower()
text = re.sub(r'[^a-zA-Z\s]', '', text)
print(text)


#### Step 3: Tokenization

# Split text into individual words.

tokens = text.split()
print(tokens)


#### Step 4: Stopword Removal / Lemmatization
# Remove common words.

stopwords = ['and', 'i']

filtered_tokens = [word for word in tokens if word not in stopwords]
print(filtered_tokens)


#### Step 5: Vocabulary Creation

# Create unique words.

vocab = list(set(filtered_tokens))
print(vocab)


# Assign IDs:

word_to_idx = {word:i for i, word in enumerate(vocab)}
print(word_to_idx)



#### Step 6: Context Window Selection

window_size = 1 # suppose

# sentence :  love nlp love machine learning

# For word nlp, neighbors are:  love ← nlp → love



#### Step 7: CBOW / Skip-Gram Pair Generation
# Skip-Gram

# Target word predicts context words.

# format : (Target , Context)

pairs = []

for i, word in enumerate(filtered_tokens):
    for j in range(max(0, i-1), min(len(filtered_tokens), i+2)):
        if i != j:
            pairs.append((word, filtered_tokens[j]))

print(pairs)



#### Step 8: Word2Vec Training

# Train the model.

# sg=1 → Skip-Gram
# sg=0 → CBOW



sentences = [filtered_tokens]

model = Word2Vec(
    sentences,
    vector_size=5,
    window=2,
    min_count=1,
    sg=1
)



#### Step 9: Word Embeddings

# Get vector for a word.

vector = model.wv['love']
print(vector)


#### Step 10: Cosine Similarity

# Compare two vectors.

similarity = model.wv.similarity('love', 'nlp')
print(similarity)


#### Step 11: Similar Words

# Find words having highest cosine similarity.

most_similar = model.wv.most_similar('love')
print(most_similar)