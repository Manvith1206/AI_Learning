from langchain.vectorstores import FAISS, faiss
import os
import pickle
from .base_vector_store import BaseVectorStore
from langchain_core.documents import Document
import infrastructure.common.RAG_Constants as constants
import numpy as np
import faiss
import uuid
import time
from infrastructure.common.component_registry import register, VECTOR_STORES_REGISTRY

@register(VECTOR_STORES_REGISTRY, name=constants.VectorStore.FAISS.value)
class FAISS_Vector_Store(BaseVectorStore):
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.db = None
        self.ids = []
        self.time_taken = 0
        self.cost = 0
        self.index_path = constants.FaissStorageConstants.FAISS_INDEX_PATH
        self.doc_path = constants.FaissStorageConstants.FAISS_DOC_PATH

    def update_index(self, index: str):
        self.index = faiss.read_index(index)
        with open(self.doc_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data.get('documents', [])
            self.ids = data.get('ids', [])
        
    def add_embeddings(self, embeddings, documents):
        start_time = time.time()
        self.documents.clear()
        if hasattr(embeddings, "toarray"):
            emb_arr = embeddings.toarray().astype(np.float32)
        else:
            emb_arr = np.array(embeddings, dtype=np.float32)

        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(1, -1)

        faiss.normalize_L2(emb_arr)

        new_ids = [str(uuid.uuid4()) for _ in documents]
        self.documents.extend(documents)
        self.ids.extend(new_ids)
        
        if self.index is None:
            dimension = emb_arr.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index.add(emb_arr)
        
        end_time = time.time()
        self.time_taken += (end_time - start_time)
        self.save_index()

    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.doc_path, 'wb') as f:
                pickle.dump({'documents': self.documents, 'ids': self.ids}, f)

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
                    constants.Constants.Document: self.documents[idx],
                    constants.Constants.Score: similarity_score,
                    constants.Constants.ID: self.ids[idx]
                })
                
        return results

    def format_documents(self, documents):
        
        return documents

    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken

    def get_index(self):
        return self.index_path
    def get_all_documents(self):
        pass