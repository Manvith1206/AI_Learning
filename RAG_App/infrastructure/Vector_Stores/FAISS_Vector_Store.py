from langchain.vectorstores import FAISS, faiss
import os
from .base_vector_store import BaseVectorStore
from langchain_core.documents import Document
import infrastructure.common.rag_constants as constants
import numpy as np
import faiss
import uuid
import time

class FAISS_Vector_Store(BaseVectorStore):
    def __init__(self):
        self.documents = None
        self.embeddings = None
        self.index = None
        self.db = None
        self.ids = None
        self.time_taken = 0
        self.cost = 0

    def update_index(self, index: faiss.IndexFlatIP):
        self.index = index
        
    def add_embeddings(self, embeddings, documents):
        start_time = time.time()
        # Unify embeddings: list, numpy array, or sparse -> dense np.float32
        self.documents = documents
        # Generate and store document IDs
        self.ids = []
        
        for document in documents:
            if isinstance(document, dict):
                doc_id = document.get("id", str(uuid.uuid4()))
            elif hasattr(document, "metadata") and isinstance(document.metadata, dict) and "id" in document.metadata:
                doc_id = document.metadata["id"]
            else:
                doc_id = str(uuid.uuid4())
            self.ids.append(doc_id)
        if hasattr(embeddings, "toarray"):
            emb_arr = embeddings.toarray().astype(np.float32)
        else:
            emb_arr = np.array(embeddings, dtype=np.float32)
        # Ensure 2D array
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(1, -1)
        dimension = emb_arr.shape[1]
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(emb_arr)
        # Build and populate FAISS index using inner product
        index = faiss.IndexFlatIP(dimension)
        index.add(emb_arr)
        self.index = index
        end_time = time.time()
        self.time_taken = end_time - start_time

    def search(self, query_embedding, top_k=5):
        """Search for most similar documents in FAISS index."""
        
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
                    constants.Score: similarity_score,
                    constants.ID: self.ids[idx]
                })
        
        self.index.docstore
        
        return results

    def format_documents(self, documents):
        
        return documents

    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken