######################Step 1: Load BERT


from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load  model
model = AutoModel.from_pretrained("bert-base-uncased")


######################Step 2: Input a sentence  (101 → [CLS] , 102 → [SEP])


sentence = "Machine learning is fascinating."

# Tokenize
inputs = tokenizer(sentence, return_tensors="pt")

print("inputs")
print(inputs)



######################Step 3: Pass through BERT model


with torch.no_grad():
    outputs = model(**inputs)
    

######################Step 4: Inspect outputs

print("outputs")
print(outputs.last_hidden_state.shape)


"""
Meaning:

1 → batch size
7 → number of tokens
768 → embedding size for each token

"""

######################Step 5: Token embeddings

token_embeddings = outputs.last_hidden_state

print("token_embeddings")
print(token_embeddings)


######################Step 6: Convert IDs back to tokens


tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for token, embedding in zip(tokens, token_embeddings[0]):
    print(token, embedding[:5])   # first 5 values
    
    
######################Step 7: Mean pooling for sentence embedding

sentence_embedding = token_embeddings.mean(dim=1)
print("Sentence embedding shape:", sentence_embedding.shape)

# Print tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("\nTokens:")
print(tokens)