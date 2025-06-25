import numpy as np
from .base_retriever import BaseRetriever
import infrastructure.common.RAG_Constants as constants
import time
from infrastructure.common.component_registry import register, RETRIEVERS_REGISTRY

@register(RETRIEVERS_REGISTRY, name=constants.RetrieverType.SIMILARITY.value)
class SimilarityRetriever(BaseRetriever):
    """Retriever that uses cosine similarity to find relevant documents"""
    
    def __init__(self, similarity_threshold=0.0, top_k = 5):
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.time_taken = 0
        self.cost = 0
    
    def retrieve(self, query_embedding, documents, **kwargs):
        """
        Retrieve documents based on vector similarity
        
        Args:
            query_embedding: Query embedding
            documents: List of document objects with embeddings
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with similarity scores
        """
        start_time = time.time()
        if hasattr(query_embedding, "toarray"):
            emb_arr = query_embedding.toarray().astype(np.float32)
        else:
            emb_arr = np.array(query_embedding, dtype=np.float32)
        # Get vector store from kwargs
        vector_store = kwargs.get(constants.ConfigManagerNames.CONFIG_VECTOR_STORE)
        if not vector_store:
            raise ValueError(constants.UIDisplayNameConstants.VECTOR_STORE_MUST_BE_PROVIDED_ERROR_MESSAGE)
            
        # Get search results from vector store
        
        results = vector_store.search(emb_arr, top_k=self.top_k)
        # Filter by similarity threshold if needed
        filtered_results = [
            result for result in results 
            if result[constants.Constants.Score] >= self.similarity_threshold
        ]
        end_time = time.time()
        self.time_taken = end_time - start_time
        final_results = [
            result[constants.Constants.Document][constants.Constants.PAGE_CONTENT] for result in filtered_results 
        ]
        return final_results
    
    def get_cost_and_time_taken(self):
        """Returns the time taken for the retrieve operation."""
        return self.cost, self.time_taken