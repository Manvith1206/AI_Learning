from langchain.vectorstores import FAISS, faiss
import os
from .base_vector_store import BaseVectorStore
from langchain_core.documents import Document
import RAG_Constants as constants
import numpy as np

import faiss

class FAISS_Vector_Store(BaseVectorStore):
    def __init__(self):
        self.documents = None
        self.embeddings = None
        self.index = None
        self.db = None

    def add_embeddings(self, embeddings, documents):
        vector_store_path = "vector_store_index_path"
        vector_store_data_path = "vector_store_data_path"
        self.documents = documents
        self.embeddings = embeddings
        breakpoint()
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            # For dense embeddings
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            index.add(embeddings)
        else:
            # Handle sparse embeddings if needed
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)

            # Convert sparse to dense
            dense_embeddings = embeddings.toarray().astype(np.float32)
            index.add(dense_embeddings)
        self.index = index
        faiss.write_index(index, vector_store_path)

    def search(self, query_embedding, top_k=5):
        """Search for most similar documents in FAISS index."""
        breakpoint()
        if self.index is None:
            raise ValueError("Vector store not initialized. Call add_embeddings first.")
        
        # Handle different input types
        if hasattr(query_embedding, "toarray"):
            query_embedding = query_embedding.toarray()
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # Ensure we have 2D array with shape (1, dim)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure we're using float32 (FAISS requirement)
        query_embedding = query_embedding.astype(np.float32)
        
        # For FAISS, search returns distances and indices
        distances, indices = self.index.search(query_embedding, k=min(top_k, len(self.documents)))
        
        # FAISS returns 2D arrays: distances[0] and indices[0] give the first row
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.documents) and idx >= 0:  # Check index bounds
                # Convert distance to similarity score
                similarity_score = 1 / (1 + distances[0][i])
                results.append({
                    constants.Document: self.documents[idx],
                    constants.Score: similarity_score
                })
        
        return results


    def format_documents(self, documents):
        
        return documents