from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """Base class for retrieval strategies"""
    
    @abstractmethod
    def retrieve(self, query_embedding, documents, **kwargs):
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
    @abstractmethod
    def get_cost_and_time_taken(self):
        """Returns the time taken for the retrieve operation."""
        pass