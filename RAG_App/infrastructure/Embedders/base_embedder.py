from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts):
        pass
    @abstractmethod
    def transform(self, texts):
        pass
    @abstractmethod
    def get_cost_and_time_taken(self):
        pass