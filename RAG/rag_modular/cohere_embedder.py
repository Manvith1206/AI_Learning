# cohere_embedder.py
from .base_embedder import BaseEmbedder
import cohere
import os
import rag_modular.RAG_Constants as constants

class CohereEmbedder(BaseEmbedder):
    def __init__(self, api_key: str = None, model: str = constants.CohereEmbedModels.COHERE_EMBED_MODEL_ENG.value):
        """
        api_key: your COHERE_API_KEY (or set via env var COHERE_API_KEY)
        model:   the Cohere embed model to use
        """
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("Cohere API key not provided. Set COHERE_API_KEY or pass api_key.")
        self.client = cohere.Client(key)
        self.model = model
        self.embeddings = None

    def fit(self, texts: list[str]) -> list[list[float]]:
        """
        For Cohere embeddings we don't need a separate fit step;
        we just embed the texts and cache if desired.
        """
        resp = self.client.embed(texts=texts, model=self.model)
        self.embeddings = resp.embeddings
        return self.embeddings

    def transform(self, texts: list[str]) -> list[list[float]]:
        """
        Embed new texts on demand.
        """
        resp = self.client.embed(texts=texts, model=self.model)
        return resp.embeddings
