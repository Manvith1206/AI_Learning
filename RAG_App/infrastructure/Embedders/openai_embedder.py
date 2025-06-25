import time
from openai import OpenAI
from .base_embedder import BaseEmbedder
import infrastructure.common.rag_constants as constants
from infrastructure.common.component_registry import register, EMBEDDERS_REGISTRY

@register(EMBEDDERS_REGISTRY, constants.EmbedderType.OPENAI.value)
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key, model="text-embedding-ada-002"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.time_taken = 0
        self.cost = 0

    def embed_documents(self, texts):
        start_time = time.time()
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [item.embedding for item in response.data]
        end_time = time.time()
        self.time_taken = end_time - start_time
        # Note: OpenAI cost calculation can be added here if needed
        return embeddings

    def transform(self, texts):
        return self.embed_documents(texts)

    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
