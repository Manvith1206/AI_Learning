from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import re
# Ensure numpy is listed in requirements.txt
import numpy as np
# Ensure rank_bm25 is listed in requirements.txt
try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    BM25Okapi = None

from app.infrastructure.vector_store.base_store import BaseVectorStore
# Using app.config.constants for key names if needed, though many are passed directly
# from app.config import constants

# Define common keys used in retrieval results, matching BaseVectorStore's search output
DOC_ID = "id"
DOC_CONTENT = "page_content" # Assuming documents have this key
DOC_SCORE = "score"
DOC_METADATA = "metadata"
DOC_OBJECT = "document" # Key for the original document object, if different from page_content

class BaseRetriever(ABC):
    def __init__(self):
        self.time_taken: float = 0.0
        self.cost: float = 0.0 # Placeholder for potential costs

    @abstractmethod
    def retrieve(
        self,
        query_embedding: List[float],
        query_text: str, # Needed by some retrievers like Hybrid
        vector_store: BaseVectorStore,
        top_k: int,
        # Optional: For retrievers like HybridBM25 that might need all docs or a way to get them
        all_document_texts: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        pass

    def get_cost_and_time_taken(self) -> tuple[float, float]:
        return self.cost, self.time_taken

class SimilarityRetriever(BaseRetriever):
    def __init__(self, similarity_threshold: float = 0.0):
        super().__init__()
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query_embedding: List[float],
        query_text: str, # Unused by basic similarity, but part of interface
        vector_store: BaseVectorStore,
        top_k: int,
        all_document_texts: Optional[List[Dict[str, Any]]] = None, # Unused
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        start_time = time.time()

        results = vector_store.search(query_embedding, top_k=top_k, filter_criteria=kwargs.get("filter_criteria"))

        filtered_results = [
            result for result in results
            if result.get(DOC_SCORE, 0.0) >= self.similarity_threshold
        ]
        self.time_taken = time.time() - start_time
        # Assuming results from vector_store.search are already in the desired format:
        # List[Dict[str, Any]] where each dict has 'page_content', 'metadata', 'score', 'id'
        return filtered_results

class HybridRetriever(BaseRetriever):
    def __init__(self, keyword_weight: float = 0.3, bm25_top_k_multiplier: int = 2):
        super().__init__()
        if not RANK_BM25_AVAILABLE:
            raise ImportError("rank_bm25 library is required for HybridRetriever.")
        self.keyword_weight = keyword_weight
        self.bm25_top_k_multiplier = bm25_top_k_multiplier # How many more docs to fetch for BM25
        self.bm25: Optional[BM25Okapi] = None
        self.doc_id_to_bm25_idx: Dict[str, int] = {}
        self.bm25_idx_to_doc_id: Dict[int, str] = {}
        self._bm25_corpus_initialized_with_ids: Optional[List[str]] = None

    def _preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def _initialize_bm25(self, documents: List[Dict[str, Any]]):
        # documents: list of dicts, each with DOC_ID and DOC_CONTENT
        doc_ids = [doc[DOC_ID] for doc in documents]

        # Avoid re-initializing if the corpus is the same
        if self._bm25_corpus_initialized_with_ids == doc_ids:
            return

        processed_corpus = []
        self.doc_id_to_bm25_idx = {}
        self.bm25_idx_to_doc_id = {}

        for i, doc in enumerate(documents):
            doc_id = doc[DOC_ID]
            text_content = doc.get(DOC_CONTENT, "")
            tokens = self._preprocess_text(text_content)
            processed_corpus.append(tokens)
            self.doc_id_to_bm25_idx[doc_id] = i
            self.bm25_idx_to_doc_id[i] = doc_id

        self.bm25 = BM25Okapi(processed_corpus)
        self._bm25_corpus_initialized_with_ids = doc_ids
        print(f"Initialized BM25 with {len(documents)} documents.")

    def retrieve(
        self,
        query_embedding: List[float],
        query_text: str,
        vector_store: BaseVectorStore,
        top_k: int,
        all_document_texts: Optional[List[Dict[str, Any]]] = None, # List of {id: '..', page_content: '...'}
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        start_time = time.time()

        if not all_document_texts:
            # This is a simplification. In a real scenario, DocumentService might need
            # to fetch all texts or have a more optimized way for BM25.
            # Or, BM25 is pre-initialized when DocumentService loads a collection.
            # For now, we'll raise an error if not provided, or one could try to fetch all.
            print("Warning: HybridRetriever expects all_document_texts for BM25 initialization if not already initialized.")
            # As a fallback, try to get all docs from vector store if possible (not a standard BaseVectorStore method)
            # For now, let's assume it must be passed or pre-initialized.
            if self.bm25 is None:
                 raise ValueError("HybridRetriever's BM25 is not initialized and all_document_texts were not provided.")
        elif self.bm25 is None or self._bm25_corpus_initialized_with_ids != [d[DOC_ID] for d in all_document_texts]:
            self._initialize_bm25(all_document_texts)

        if self.bm25 is None: # Should not happen if logic above is correct
            raise RuntimeError("BM25 model not available in HybridRetriever.")

        # 1. Semantic Search (from vector store)
        # Fetch more results initially for better hybrid combination, e.g., top_k * 2
        semantic_results_list = vector_store.search(query_embedding, top_k=top_k * self.bm25_top_k_multiplier, filter_criteria=kwargs.get("filter_criteria"))

        semantic_results_map: Dict[str, Dict[str, Any]] = {res[DOC_ID]: res for res in semantic_results_list}

        # 2. BM25 Keyword Search
        processed_query_tokens = self._preprocess_text(query_text)
        # bm25_scores_all_docs: np.ndarray (scores for each doc in BM25 corpus)
        bm25_scores_all_docs = self.bm25.get_scores(processed_query_tokens)

        max_bm25_score = np.max(bm25_scores_all_docs) if bm25_scores_all_docs.size > 0 else 1.0
        if max_bm25_score == 0: max_bm25_score = 1.0 # Avoid division by zero

        # Combine scores
        combined_results: List[Dict[str, Any]] = []

        # Consider all docs that appeared in semantic results or have a BM25 score
        # This ensures we consider docs found by either method.
        relevant_doc_ids = set(semantic_results_map.keys())
        for bm25_idx, bm25_score_val in enumerate(bm25_scores_all_docs):
            if bm25_score_val > 0: # Only consider docs with some keyword match
                 relevant_doc_ids.add(self.bm25_idx_to_doc_id[bm25_idx])

        for doc_id in relevant_doc_ids:
            semantic_doc = semantic_results_map.get(doc_id)

            semantic_score = semantic_doc.get(DOC_SCORE, 0.0) if semantic_doc else 0.0

            bm25_idx = self.doc_id_to_bm25_idx.get(doc_id)
            keyword_score = 0.0
            if bm25_idx is not None:
                keyword_score = bm25_scores_all_docs[bm25_idx] / max_bm25_score # Normalized

            combined_score = ((1 - self.keyword_weight) * semantic_score) + (self.keyword_weight * keyword_score)

            # Construct the result document. If not in semantic_results, need to fetch its content.
            # This part is tricky if vector_store.search is the only way to get doc content.
            # Assuming semantic_doc contains content if available, or a placeholder.
            # For simplicity, if a doc is only found by BM25, we'd need a way to get its content.
            # This implies 'all_document_texts' should be readily available or BaseVectorStore needs a get_by_id.

            doc_data_to_return = semantic_doc
            if not doc_data_to_return and all_document_texts: # Doc found by BM25 but not semantic
                original_doc = next((d for d in all_document_texts if d[DOC_ID] == doc_id), None)
                if original_doc:
                     doc_data_to_return = {**original_doc, DOC_SCORE: 0.0} # placeholder semantic score
                else: # Should not happen if doc_id came from bm25_idx_to_doc_id
                    continue

            if not doc_data_to_return: # Still no data
                continue

            final_doc_obj = {
                **doc_data_to_return, # Includes original id, page_content, metadata
                DOC_SCORE: combined_score, # Overwrite with combined score
                "semantic_score_component": semantic_score, # For transparency
                "keyword_score_component": keyword_score   # For transparency
            }
            combined_results.append(final_doc_obj)

        combined_results.sort(key=lambda x: x[DOC_SCORE], reverse=True)
        self.time_taken = time.time() - start_time
        return combined_results[:top_k]

def get_retriever(retriever_type: str, params: Optional[Dict[str, Any]] = None) -> BaseRetriever:
    params = params or {}
    if retriever_type == "similarity":
        return SimilarityRetriever(**params)
    elif retriever_type == "hybrid":
        return HybridRetriever(**params)
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")
