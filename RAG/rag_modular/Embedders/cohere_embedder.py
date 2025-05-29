# cohere_embedder.py
import time
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
        self.cost = 0
        self.time_taken = 0

    def batch_chunks(self, chunks, batch_size=80):
        """Yield successive batches of size batch_size."""
        for i in range(0, len(chunks), batch_size):
            print("batching is in progress")
            yield chunks[i:i + batch_size]

    def fit(self, texts: list[str]):
        """
        For Cohere embeddings we don't need a separate fit step;
        we just embed the texts and cache if desired.
        """
        start_time = time.time()
        self.texts = texts
        all_embeddings = []
        current_cost_value = 0
        for batch in self.batch_chunks(texts, batch_size=80): # Adjust batch_size as needed for Cohere
            resp = self.client.embed(texts=batch, model=self.model, input_type="search_document")
            # if resp.meta and resp.meta.billed_units and resp.meta.billed_units.input_tokens is not None:
            #     current_cost_value += self.get_cost_based_on_model(resp.meta.billed_units.input_tokens)
            # else:
            #     print("Warning: Cohere API response did not include input_tokens. Cost metric might be inaccurate.")
            all_embeddings.extend(resp.embeddings)
        self.embeddings = all_embeddings
        end_time = time.time()
        self.time_taken = end_time - start_time
        self.cost = current_cost_value 
        return self.embeddings

    def transform(self, texts: list[str]):
        """
        Embed new texts on demand.
        """
        all_embeddings = []
        for batch in self.batch_chunks(texts, batch_size=80): # Adjust batch_size as needed for Cohere
            resp = self.client.embed(texts=batch, model=self.model)
            all_embeddings.extend(resp.embeddings)
        return all_embeddings
    
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
    
    def get_cost_based_on_model(self, tokens):
        if self.model == constants.CohereEmbedModels.COHERE_EMBED_MODEL_ENG.value:
            return (tokens/1000) * 0.0001
        elif self.model == constants.CohereEmbedModels.COHERE_EMBED_MODEL_DEFAULT.value:
            return (tokens / 1000000) * 0.12