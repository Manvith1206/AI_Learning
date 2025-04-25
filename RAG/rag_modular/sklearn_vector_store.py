from sklearn.neighbors import NearestNeighbors
import numpy as np
from .base_vector_store import BaseVectorStore
import rag_modular.RAG_Constants as constants

class SklearnVectorStore(BaseVectorStore):
    def __init__(self, metric='cosine'):
        self.metric = metric
        self.nn_model = None
        self.documents = None
        self.embeddings = None

    def add_embeddings(self, embeddings, documents):
        # assign documents and log for debugging
        self.documents = documents
        print("SklearnVectorStore.documents set to:")
        print(self.documents[0])
        if hasattr(embeddings, "toarray"):
            embeddings = embeddings.toarray()
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        self.embeddings = embeddings
        n_neighbors = min(5, len(documents))
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        self.nn_model.fit(embeddings)
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
                constants.Score: similarity_scores[i]
            })
        return results
