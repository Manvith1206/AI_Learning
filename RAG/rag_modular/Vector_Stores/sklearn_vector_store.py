from sklearn.neighbors import NearestNeighbors
import numpy as np
from .base_vector_store import BaseVectorStore
import rag_modular.Common.RAG_Constants as constants
import uuid
import time

class SklearnVectorStore(BaseVectorStore):
    def __init__(self, metric='cosine'):
        self.metric = metric
        self.nn_model = None
        self.documents = None
        self.embeddings = None
        self.ids = None
        self.time_taken = 0
        self.cost = 0

    def add_embeddings(self, embeddings, documents):
        start_time = time.time()
        # assign documents and generate IDs
        self.documents = documents
        self.ids = []
        if not documents: # If there are no documents, there's nothing to add or embed.
            self.embeddings = np.array([]) # Store empty numpy array for consistency
            self.nn_model = None # Ensure model is not fitted
            print("Warning: No documents provided to SklearnVectorStore.add_embeddings. Store will be empty.")
            return

        for document in documents:
            if isinstance(document, dict):
                doc_id = document.get("id", str(uuid.uuid4()))
            elif hasattr(document, "metadata") and isinstance(document.metadata, dict) and "id" in document.metadata:
                doc_id = document.metadata["id"]
            else:
                doc_id = str(uuid.uuid4())
            self.ids.append(doc_id)
        # Unwrap wrapper objects (e.g. ContentEmbedding) into raw floats
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if hasattr(first, "values"):
                embeddings = [e.values for e in embeddings]
            elif hasattr(first, "embedding"):
                embeddings = [e.embedding for e in embeddings]
        # Convert to numpy array
        if hasattr(embeddings, "toarray"):
            embeddings = embeddings.toarray()
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        
        # Check if embeddings are empty after processing
        if embeddings.size == 0:
            self.embeddings = np.array([]) # Store empty numpy array
            self.nn_model = None # No model to fit if no embeddings
            print("Warning: Embeddings list was empty or resulted in an empty array. SklearnVectorStore will be empty.")
            return

        # store embeddings and fit model
        self.embeddings = embeddings
        # n_neighbors should be at least 1 if embeddings is not empty, and not more than num samples.
        n_neighbors = min(max(1, len(documents)), 5) 
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        self.nn_model.fit(embeddings)
        end_time = time.time()
        self.time_taken = end_time - start_time
        print("Add Embeddings Succesfull")

    def search(self, query_embedding, top_k=5):
        if self.nn_model is None:
            raise ValueError("Vector store not initialized. Call add_embeddings first.")
        if hasattr(query_embedding, "toarray"):
            query_embedding = query_embedding.toarray()
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        distances, indices = self.nn_model.kneighbors(query_embedding, n_neighbors=min(top_k, len(self.documents)))
        similarity_scores = 1 - distances.flatten()
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                constants.Document: self.documents[idx],
                constants.Score: similarity_scores[i],
                constants.ID: self.ids[idx]
            })
        return results

    def format_documents(self, documents):
        return documents
    
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
