from .base_retriever import BaseRetriever
import infrastructure.common.RAG_Constants as constants
import time
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any
from infrastructure.common.component_registry import register, RETRIEVERS_REGISTRY

@register(RETRIEVERS_REGISTRY, name=constants.RetrieverType.HYBRID.value)
class HybridRetriever(BaseRetriever):
    """
    Retriever that combines BM25 keyword scoring with semantic search
    for better hybrid document retrieval.
    """
    
    def __init__(self, keyword_weight=0.3, top_k=5):
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        self.time_taken = 0
        self.cost = 0
        self.bm25 = None
        self.doc_mapping = {}  # Maps BM25 corpus index to document ID
        
    def _preprocess_text(self, text: str) -> str:
        """Clean and tokenize text"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text
    
    def _initialize_bm25(self, documents: List[Dict[str, Any]]):
        """Initialize BM25 with the document corpus"""
        # Preprocess documents
        processed_docs = []
        self.doc_mapping = {}
        
        for idx, doc in enumerate(documents):
            text = self._preprocess_text(doc[constants.Constants.PAGE_CONTENT])
            tokens = text.split()
            processed_docs.append(tokens)
            self.doc_mapping[idx] = doc[constants.Constants.ID]
            
        # Initialize BM25
        self.bm25 = BM25Okapi(processed_docs)

    def retrieve(self, query_embedding, documents, **kwargs):
        start_time = time.time()
        
        # Get vector store and query text from kwargs
        vector_store = kwargs.get(constants.ConfigManagerNames.CONFIG_VECTOR_STORE)
        if not vector_store:
            raise ValueError(constants.UIDisplayNameConstants.VECTOR_STORE_MUST_BE_PROVIDED_ERROR_MESSAGE)
            
        query_text = kwargs.get(constants.Constants.QUERY_TEXT)
        if not query_text:
            raise ValueError(constants.UIDisplayNameConstants.QUERY_TEXT_MUST_BE_PROVIDED_ERROR_MESSAGE)
        
        # Initialize BM25 if not already done
        if self.bm25 is None:
            self._initialize_bm25(documents)
        
        # Get semantic search results
        semantic_results = vector_store.search(query_embedding, top_k=self.top_k*2)
        
        # Get BM25 scores
        processed_query = self._preprocess_text(query_text)
        query_tokens = processed_query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to [0,1] range
        max_bm25_score = max(bm25_scores) if bm25_scores.any() else 1
        normalized_bm25_scores = {
            self.doc_mapping[idx]: score/max_bm25_score 
            for idx, score in enumerate(bm25_scores)
        }
        
        # Combine scores
        combined_results = []
        for result in semantic_results:
            doc_id = result[constants.Constants.ID]
            semantic_score = result[constants.Constants.Score]
            keyword_score = normalized_bm25_scores.get(doc_id, 0)
            
            # Weighted combination
            combined_score = (
                (1 - self.keyword_weight) * semantic_score + 
                self.keyword_weight * keyword_score
            )
            
            combined_results.append({
                constants.Constants.Document: result[constants.Constants.Document],
                constants.Constants.ID: result[constants.Constants.ID],
                constants.Constants.Score: combined_score,
                constants.Constants.SEMANTIC_SCORE: semantic_score,
                constants.Constants.KEYWORD_SCORE: keyword_score
            })
        
        # Sort by combined score and take top_k
        combined_results.sort(key=lambda x: x[constants.Constants.Score], reverse=True)
        end_time = time.time()
        self.time_taken = end_time - start_time
        
        final_results = []
        for result in combined_results[:self.top_k]:
            final_results.append(result[constants.Constants.Document][constants.Constants.PAGE_CONTENT])

        return final_results

    def get_cost_and_time_taken(self):
        """Returns the time taken for the retrieve operation."""
        return self.cost, self.time_taken
