import numpy as np
from .base_retriever import BaseRetriever
import rag_modular.Common.RAG_Constants as constants

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
        vector_store = kwargs.get(constants.CONFIG_VECTOR_STORE)
        if not vector_store:
            raise ValueError(constants.VECTOR_STORE_MUST_BE_PROVIDED_ERROR_MESSAGE)
            
        # Get search results from vector store
        results = vector_store.search(query_embedding, top_k=top_k)
        
        # Filter by similarity threshold if needed
        filtered_results = [
            result for result in results 
            if result[constants.Score] >= self.similarity_threshold
        ]
        
        return filtered_results
