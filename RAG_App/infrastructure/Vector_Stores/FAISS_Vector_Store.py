from langchain_core.documents import Document
import numpy as np
import faiss
import os
import pickle
import time
import uuid
from .base_vector_store import BaseVectorStore
import infrastructure.common.rag_constants as constants

class FAISS_Vector_Store(BaseVectorStore):
    def __init__(self, index_path=constants.FAISS_INDEX_PATH, documents_path=constants.FAISS_DOC_PATH):
        self.index_path = index_path
        self.documents_path = documents_path
        self.index = None
        self.documents = []
        self.ids = []
        self.time_taken = 0
        self.cost = 0
        self.load_index()

    def add_embeddings(self, embeddings, documents):
        start_time = time.time()

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

    def search(self, query_embedding, top_k=5):
        if self.index is None or self.index.ntotal == 0:
            return []

        if hasattr(query_embedding, "toarray"):
            query_embedding = query_embedding.toarray()
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k=k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if 0 <= idx < len(self.documents):
                # For IndexFlatIP, distance is the inner product, which is the similarity score
                similarity_score = distances[0][i]
                results.append({
                    constants.Document: self.documents[idx],
                    constants.Score: float(similarity_score),
                    constants.ID: self.ids[idx]
                })
        
        return results

    def format_documents(self, documents):
        return [doc.page_content for doc in documents]

    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.documents_path, 'wb') as f:
                pickle.dump({'documents': self.documents, 'ids': self.ids}, f)

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.documents_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.ids = data.get('ids', [])
            except (IOError, pickle.UnpicklingError, EOFError) as e:
                # If loading fails, start with a fresh index
                self.index = None
                self.documents = []
                self.ids = []

    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken