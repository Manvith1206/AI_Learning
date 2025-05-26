from .base_reranker import BaseReranker
from sklearn.metrics.pairwise import cosine_similarity
import time
from rag_modular.Common.RAG_Constants import RERANK_EXPLAINATION

class CosineReranker(BaseReranker):
    """
    Reranker that orders document chunks by cosine similarity with the query embedding.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self.time_taken = 0
        self.cost = 0

    def rerank(self, query, documents, **kwargs):
        start_time = time.time()
        # Compute embeddings
        
        query_vec = self.embedder.transform([query])  # shape (1, dim)
        doc_vecs = self.embedder.transform(documents)  # shape (n, dim)
        
        # Compute cosine similarities
        sims = cosine_similarity(doc_vecs, query_vec).flatten()
        # Pair docs with scores and sort
        paired = list(zip(documents, sims))
        paired.sort(key=lambda x: x[1], reverse=True)
        sorted_docs = [doc for doc, score in paired]
        # Explanation
        explanation = RERANK_EXPLAINATION
        end_time = time.time()
        self.time_taken = end_time - start_time
        return sorted_docs, explanation

    def get_cost_and_time_taken(self):
        """Returns the time taken for the rerank operation."""
        return self.cost, self.time_taken
