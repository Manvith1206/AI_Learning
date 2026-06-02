from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embedder import BaseEmbedder

class TFIDFEmbedder(BaseEmbedder):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
    def fit(self, texts):
        self.vectors = self.vectorizer.fit_transform(texts)
        return self.vectors
    def transform(self, texts):
        # Ensure fit() has been called
        
        if self.vectors is None:
            raise ValueError("TF-IDF Embedder not fitted. Please process documents (fit) before querying.")
        return self.vectorizer.transform(texts)
