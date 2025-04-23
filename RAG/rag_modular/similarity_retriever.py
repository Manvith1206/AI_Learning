import numpy as np
from .base_retriever import BaseRetriever

class SimilarityRetriever(BaseRetriever):
    """Retriever that uses cosine similarity to find relevant documents"""
    
    def __init__(self, similarity_threshold=0.0):
        self.similarity_threshold = similarity_threshold
    
    def retrieve(self, query_embedding, documents, top_k=5, **kwargs):
        """
        Retrieve documents based on vector similarity
        
        Args:
            query_embedding: Query embedding
            documents: List of document objects with embeddings
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with similarity scores
        """
        # Get vector store from kwargs
        vector_store = kwargs.get('vector_store')
        if not vector_store:
            raise ValueError("Vector store must be provided")
            
        # Get search results from vector store
        results = vector_store.search(query_embedding, top_k=top_k)
        print("Results retrieved successfully")
        
        # Filter by similarity threshold if needed
        filtered_results = [
            result for result in results 
            if result["score"] >= self.similarity_threshold
        ]
        print("Filtered results successfully")
        print(filtered_results)
        
        return filtered_results
