from .base_retriever import BaseRetriever

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
        vector_store = kwargs.get('vector_store')
        if not vector_store:
            raise ValueError("Vector store must be provided")
            
        # Get query text from kwargs
        query_text = kwargs.get('query_text')
        if not query_text:
            raise ValueError("Query text must be provided")
            
        # Get semantic search results
        semantic_results = vector_store.search(query_embedding, top_k=top_k*2)
        
        # Simple keyword matching (as a basic example)
        # In a real implementation, you might use BM25 or another keyword algorithm
        keyword_scores = {}
        query_terms = set(query_text.lower().split())
        
        for i, doc in enumerate(documents):
            content = doc["page_content"].lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in content)
            # Normalize by query length
            score = matches / max(1, len(query_terms))
            keyword_scores[doc["id"]] = score
        
        # Combine scores
        combined_results = []
        for result in semantic_results:
            doc_id = result["document"]["id"]
            semantic_score = result["score"]
            keyword_score = keyword_scores.get(doc_id, 0)
            
            # Weighted combination
            combined_score = (
                (1 - self.keyword_weight) * semantic_score + 
                self.keyword_weight * keyword_score
            )
            
            combined_results.append({
                "document": result["document"],
                "score": combined_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score
            })
        
        # Sort by combined score and take top_k
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]
