import time
import voyageai
import voyageai.client
from .base_embedder import BaseEmbedder
import rag_modular.Common.RAG_Constants as constants

class VoyageEmbedder(BaseEmbedder):
    def __init__(self, api_key, model):
        """
        api_key: your COHERE_API_KEY (or set via env var COHERE_API_KEY)
        model:   the Cohere embed model to use
        """ 
        key = api_key
        self.client = voyageai.Client(api_key=key)
        self.model = model
        self.embeddings = [] # Initialize as empty list for batching
        self.time_taken = 0
        self.cost = 0
        
    def batch_chunks(self, chunks, batch_size=80):
        """Yield successive batches of size batch_size."""
        for i in range(0, len(chunks), batch_size):
            print("batching is in progress")
            yield chunks[i:i + batch_size]
        
    def fit(self, texts):
        """
        For Voyage embeddings we don't need a separate fit step;
        we just embed the texts and cache if desired.
        """ 
        start_time = time.time()
        self.texts = texts
        all_embeddings = []
        for batch in self.batch_chunks(texts, batch_size=80): # Adjust batch_size as needed for Voyage
            emb = self.client.embed(texts=batch, model=self.model)
            all_embeddings.extend(emb.embeddings)
        self.embeddings = all_embeddings
        end_time = time.time()
        self.time_taken = end_time - start_time

        return self.embeddings
    def transform(self, texts):
        """
        Embed new Texts on demand
        """ 
        start_time = time.time()

        all_embeddings = []
        current_cost_value = 0
        for batch in self.batch_chunks(texts, batch_size=80): # Adjust batch_size as needed for Voyage
            resp = self.client.embed(texts=batch, model=self.model)
            current_cost_value += self.get_cost_based_on_model(resp.total_tokens)
            all_embeddings.extend(resp.embeddings)
        end_time = time.time()
        self.time_taken += end_time - start_time
        self.cost = current_cost_value
        return all_embeddings
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
    
    def get_cost_based_on_model(self, tokens):
        if self.model == constants.VoyageEmbedModels.VOYAGE_EMBED_DEFAULT_MODEL.value:
            return (tokens / 1000000) * 0.18
        elif self.model == constants.VoyageEmbedModels.VOYAGE_CODE_2_EMBED_MODEL.value:
            return (tokens / 1000000) * 0.12
        elif self.model == constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_5.value:
            return (tokens / 1000000) * 3
        elif self.model == constants.CLAUDE_MODELS.CLAUDE_OPUS_THREE.value:
            return (tokens / 1000000) * 15
        else:
            return 0