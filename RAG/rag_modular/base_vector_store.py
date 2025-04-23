from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    @abstractmethod
    def add_embeddings(self, embeddings, documents):
        pass
    @abstractmethod
    def search(self, query_embedding, top_k=5):
        pass
