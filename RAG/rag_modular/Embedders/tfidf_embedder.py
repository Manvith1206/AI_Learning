import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from .base_embedder import BaseEmbedder

class TFIDFEmbedder(BaseEmbedder):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.time_taken = 0
        self.cost = 0
    def fit(self, texts):
        start_time = time.time()
        self.vectors = self.vectorizer.fit_transform(texts)
        end_time = time.time()
        self.time_taken = end_time - start_time
        return self.vectors
    def transform(self, texts):
        # Ensure fit() has been called
        start_time = time.time()
        
        if self.vectors is None:
            raise ValueError("TF-IDF Embedder not fitted. Please process documents (fit) before querying.")
        vectors = self.vectorizer.transform(texts)
        end_time = time.time()
        self.time_taken += end_time - start_time
        
        return vectors
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken