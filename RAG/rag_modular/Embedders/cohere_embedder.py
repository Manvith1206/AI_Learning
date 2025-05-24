# cohere_embedder.py
from rag_modular.Embedders.base_embedder import BaseEmbedder
import cohere
import os
import rag_modular.Common.RAG_Constants as constants

class CohereEmbedder(BaseEmbedder):
    def __init__(self, api_key: str = None, model: str = constants.CohereEmbedModels.COHERE_EMBED_MODEL_ENG):
        """
        api_key: your COHERE_API_KEY (or set via env var COHERE_API_KEY)
        model:   the Cohere embed model to use
        """
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("Cohere API key not provided. Set COHERE_API_KEY or pass api_key.")
        self.client = cohere.Client(key)
        self.model = model
        self.embeddings = [] # Initialize as empty list for batching

    def batch_chunks(self, chunks, batch_size=80):
        """Yield successive batches of size batch_size."""
        for i in range(0, len(chunks), batch_size):
            print("batching is in progress")
            yield chunks[i:i + batch_size]

    def fit(self, texts: list[str]) -> list[list[float]]:
        """
        For Cohere embeddings we don't need a separate fit step;
        we just embed the texts and cache if desired.
        """
        self.texts = texts
        all_embeddings = []
        for batch in self.batch_chunks(texts, batch_size=80): # Adjust batch_size as needed for Cohere
            resp = self.client.embed(texts=batch, model=self.model)
            all_embeddings.extend(resp.embeddings)
        self.embeddings = all_embeddings
        return self.embeddings

    def transform(self, texts: list[str]) -> list[list[float]]:
        """
        Embed new texts on demand.
        """
        all_embeddings = []
        for batch in self.batch_chunks(texts, batch_size=80): # Adjust batch_size as needed for Cohere
            resp = self.client.embed(texts=batch, model=self.model)
            all_embeddings.extend(resp.embeddings)
        return all_embeddings