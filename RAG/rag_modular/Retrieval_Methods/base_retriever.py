from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """Base class for retrieval strategies"""
    
    @abstractmethod
    def retrieve(self, query_embedding, documents, top_k=5, **kwargs):
        """
        Retrieve relevant documents based on query embedding
        
        Args:
            query_embedding: The embedding of the query
            documents: List of documents to search through
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with relevance scores
        """
        pass
