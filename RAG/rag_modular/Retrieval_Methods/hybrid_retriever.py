from .base_retriever import BaseRetriever
import rag_modular.Common.RAG_Constants as constants
import time

class HybridRetriever(BaseRetriever):
    """
    Retriever that combines multiple retrieval strategies
    (e.g., keyword + semantic search)
    """
    
    def __init__(self, keyword_weight=0.3, top_k = 5):
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        self.time_taken = 0
        self.cost = 0
        
    def retrieve(self, query_embedding, documents, **kwargs):
        start_time = time.time()
        """
        Retrieve documents using a hybrid approach
        
        Args:
            query_embedding: Query embedding
            documents: List of document objects
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with combined scores
        """
        # Get vector store from kwargs
        vector_store = kwargs.get(constants.CONFIG_VECTOR_STORE)
        if not vector_store:
            raise ValueError(constants.VECTOR_STORE_MUST_BE_PROVIDED_ERROR_MESSAGE)
            
        # Get query text from kwargs
        query_text = kwargs.get(constants.QUERY_TEXT)
        if not query_text:
            raise ValueError(constants.QUERY_TEXT_MUST_BE_PROVIDED_ERROR_MESSAGE)
            
        # Get semantic search results
        semantic_results = vector_store.search(query_embedding, top_k=self.top_k*2)
        
        # Simple keyword matching (as a basic example)
        # In a real implementation, you might use BM25 or another keyword algorithm
        keyword_scores = {}
        query_terms = set(query_text.lower().split())
        
        for i, doc in enumerate(documents):
            content = doc[constants.PAGE_CONTENT].lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in content)
            # Normalize by query length
            score = matches / max(1, len(query_terms))
            
            keyword_scores[doc[constants.ID]] = score
        
        # Combine scores
        combined_results = []
        for result in semantic_results:
            
            doc_id = result[constants.ID]
            semantic_score = result[constants.Score]
            keyword_score = keyword_scores.get(doc_id, 0)
            
            # Weighted combination
            combined_score = (
                (1 - self.keyword_weight) * semantic_score + 
                self.keyword_weight * keyword_score
            )
            
            combined_results.append({
                constants.Document: result[constants.Document],
                constants.ID: result[constants.ID],
                constants.Score: combined_score,
                constants.SEMANTIC_SCORE: semantic_score,
                constants.KEYWORD_SCORE: keyword_score
            })
        
        # Sort by combined score and take top_k
        combined_results.sort(key=lambda x: x[constants.Score], reverse=True)
        end_time = time.time()
        self.time_taken = end_time - start_time
        return combined_results[:self.top_k]

    def get_cost_and_time_taken(self):
        """Returns the time taken for the retrieve operation."""
        return self.cost, self.time_taken
