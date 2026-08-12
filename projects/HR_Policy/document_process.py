import json
from pathlib import Path

from src.countvectorizer_tfidf import CountVectorizerTfidfEmbedding
from src.word2vec import Word2VecProcess
from src.sentencetransformers import SentenceTransformerProcess
from src.open_ai import OpenAIProcess

class DocumentProcess:

    def __init__(self, similarity_metric, embedding_method, top_k, question, openai_api_key=None):
        
        self.similarity_metric = similarity_metric
        self.embedding_method = embedding_method
        self.top_k = top_k
        self.question = question
        self.openai_api_key = openai_api_key

    def process(self):

        print("Similarity Metric:", self.similarity_metric)
        print("Embedding Method:", self.embedding_method)
        print("Top K:", self.top_k)
        print("Question:", self.question)
        print("OpenAI API Key:", self.openai_api_key)
        
        
            
        BASE_DIR = Path(__file__).resolve().parent
        print("BASE_DIR : ", BASE_DIR)
        DATA_PATH = BASE_DIR / "data" / "hr_policies.json"

        # ============================================================
        # Load Data
        # ============================================================
        raw_data = self.load_policy_data(DATA_PATH)
        
        
        if self.embedding_method in ["CountVectorizer", "TF-IDF"]:
            similarity_Score, Answer = CountVectorizerTfidfEmbedding(raw_data=raw_data, similarity_metric = self.similarity_metric, embedding_method = self.embedding_method,
                                               top_k = self.top_k, user_query = self.question).generate_search_engine()
            
            # print("=====Answer====== ", Answer)
            
        elif self.embedding_method in ["Word2Vec"]:
            similarity_Score, Answer = Word2VecProcess(raw_data=raw_data, similarity_metric = self.similarity_metric, embedding_method = self.embedding_method,
                                                       top_k = self.top_k, user_query = self.question).generate_search_engine()
                    
        elif self.embedding_method in ["SentenceTransformer"]:
            similarity_Score, Answer = SentenceTransformerProcess(raw_data=raw_data, similarity_metric = self.similarity_metric, embedding_method = self.embedding_method,
                                                               top_k = self.top_k, user_query = self.question).generate_search_engine()
                            
        elif self.embedding_method in ["OpenAI"]:
            similarity_Score, Answer = OpenAIProcess(raw_data=raw_data, similarity_metric = self.similarity_metric, embedding_method = self.embedding_method,
                                                                       top_k = self.top_k, user_query = self.question, openai_api_key = self.openai_api_key).generate_search_engine()
        
        return similarity_Score, Answer
    
    def load_policy_data(self,path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)