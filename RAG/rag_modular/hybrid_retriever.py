from .base_retriever import BaseRetriever
import rag_modular.RAG_Constants as constants

class HybridRetriever(BaseRetriever):
    """
    Retriever that combines multiple retrieval strategies
    (e.g., keyword + semantic search)
    """
    
    def __init__(self, keyword_weight=0.3):
        self.keyword_weight = keyword_weight
        
    def retrieve(self, query_embedding, documents, top_k=5, **kwargs):
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
        semantic_results = vector_store.search(query_embedding, top_k=top_k*2)
        
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
        return combined_results[:top_k]
