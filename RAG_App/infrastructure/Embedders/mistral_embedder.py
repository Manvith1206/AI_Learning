import time
import logging
from typing import List, Tuple
from mistralai.client import MistralClient
from mistralai.models.embeddings import EmbeddingResponse
from .base_embedder import BaseEmbedder
from ..common.component_registry import EMBEDDERS_REGISTRY
import infrastructure.common.rag_constants as constants

logger = logging.getLogger(__name__)

@EMBEDDERS_REGISTRY.register(constants.EmbedderType.MISTRAL.value)
class MistralEmbedder(BaseEmbedder):
    """An embedder that uses the Mistral API to generate text embeddings."""

    def __init__(self, api_key: str, model: str = "mistral-embed", batch_size: int = 32):
        """
        Initializes the MistralEmbedder.

        Args:
            api_key (str): The Mistral API key.
            model (str): The Mistral embedding model to use.
            batch_size (int): The number of documents to process in a single batch.
        """
        if not api_key:
            raise ValueError("Mistral API key is required.")
        
        self.client = MistralClient(api_key=api_key)
        self.model_name = model
        self.batch_size = batch_size
        self._time_taken = 0.0
        self._cost = 0.0

    def _batch_chunks(self, texts: List[str]) -> List[List[str]]:
        """Yields successive batches of a specified size."""
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents using the Mistral API.

        Args:
            texts (List[str]): A list of documents to embed.

        Returns:
            List[List[float]]: A list of embeddings.
        """
        start_time = time.time()
        if not texts:
            self._time_taken = time.time() - start_time
            self._cost = 0.0
            return []

        all_embeddings = []
        total_tokens = 0

        for batch in self._batch_chunks(texts):
            try:
                response: EmbeddingResponse = self.client.embeddings(
                    model=self.model_name,
                    input=batch
                )
                all_embeddings.extend([d.embedding for d in response.data])
                total_tokens += response.usage.total_tokens
            except Exception as e:
                logger.error(f"Error embedding batch with Mistral: {e}")
                all_embeddings.extend([[]] * len(batch))

        self._cost = self._get_cost(total_tokens)
        self._time_taken = time.time() - start_time
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query using the Mistral API.

        Args:
            query (str): The query string to embed.

        Returns:
            List[float]: The embedding for the query.
        """
        start_time = time.time()
        if not query:
            self._time_taken = time.time() - start_time
            self._cost = 0.0
            return []

        try:
            response: EmbeddingResponse = self.client.embeddings(
                model=self.model_name,
                input=[query]
            )
            embedding = response.data[0].embedding
            self._cost = self._get_cost(response.usage.total_tokens)
        except Exception as e:
            logger.error(f"Error embedding query with Mistral: {e}")
            embedding = []
            self._cost = 0.0

        self._time_taken = time.time() - start_time
        return embedding

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the cost and time taken for the last embedding operation.
        """
        return self._cost, self._time_taken

    def _get_cost(self, num_tokens: int) -> float:
        """
        Calculates the cost based on the number of tokens.
        Pricing: $0.1 / 1M tokens
        """
        return (num_tokens / 1_000_000) * 0.10