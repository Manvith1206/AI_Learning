from mistralai import Mistral
from .base_embedder import BaseEmbedder
import infrastructure.common.rag_constants as constants
import random, time, logging
from infrastructure.common.component_registry import register, EMBEDDERS_REGISTRY

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

@register(EMBEDDERS_REGISTRY, constants.EmbedderType.MISTRAL.value)
class MistralEmbedder(BaseEmbedder):
    """Mistral embedding model implementation with batching and rate limiting"""
    def __init__(self, api_key, model="mistral-embed", batch_size=20):
        """Initialize the Mistral embedding model"""
        api_key = api_key
        if not api_key:
            raise ValueError("Mistral API key not found in environment variables")
        
        self.client = Mistral(api_key=api_key)
        self.model_name = model
        self.batch_size = batch_size
        self.time_taken = 0
        self.cost = 0
    
    def batch_chunks(self, chunks, batch_size):
        """Yield successive batches of size batch_size."""
        
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def transform(self, texts):
        self.embed_documents(texts)

    def embed_documents(self, texts):
        all_embeddings = []
        start_time = time.time()

        for text_batch in self.batch_chunks(texts, batch_size=self.batch_size): 
            if not text_batch:
                continue

            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    inputs=text_batch
                )
                
            except Exception as e:
                print(f"Error embedding batch: {e}")
                continue 

            current_batch_extracted_values = []
            for resp in response.data:
                current_batch_extracted_values = resp.embedding

            all_embeddings.extend(current_batch_extracted_values)

        
        end_time = time.time()
        self.time_taken = end_time - start_time
        return all_embeddings

    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken