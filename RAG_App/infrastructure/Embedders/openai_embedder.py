import time
import logging
from typing import List, Tuple
from openai import OpenAI
from .base_embedder import BaseEmbedder
from ..common.component_registry import EMBEDDERS_REGISTRY
import infrastructure.common.RAG_Constants as constants

logger = logging.getLogger(__name__)

@EMBEDDERS_REGISTRY.register(constants.EmbedderType.OPENAI.value)
class OpenAIEmbedder(BaseEmbedder):
    """An embedder that uses the OpenAI API to generate text embeddings."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", batch_size: int = 32):
        """
        Initializes the OpenAIEmbedder.

        Args:
            api_key (str): The OpenAI API key.
            model (str): The OpenAI embedding model to use.
            batch_size (int): The number of documents to process in a single batch.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self._time_taken = 0.0
        self._cost = 0.0

    def _batch_chunks(self, texts: List[str]) -> List[List[str]]:
        """Yields successive batches of a specified size."""
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents using the OpenAI API.

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
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                all_embeddings.extend([item.embedding for item in response.data])
                total_tokens += response.usage.total_tokens
            except Exception as e:
                logger.error(f"Error embedding batch with OpenAI: {e}")
                all_embeddings.extend([[]] * len(batch))

        self._cost = self._get_cost(total_tokens)
        self._time_taken = time.time() - start_time
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query using the OpenAI API.

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
            response = self.client.embeddings.create(
                input=[query],
                model=self.model
            )
            embedding = response.data[0].embedding
            self._cost = self._get_cost(response.usage.total_tokens)
        except Exception as e:
            logger.error(f"Error embedding query with OpenAI: {e}")
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
        Calculates the cost based on the number of tokens for the specific model.
        Pricing for text-embedding-3-small: $0.02 / 1M tokens
        Pricing for text-embedding-3-large: $0.13 / 1M tokens
        Pricing for text-embedding-ada-002: $0.10 / 1M tokens
        """
        if self.model == "text-embedding-3-large":
            return (num_tokens / 1_000_000) * 0.13
        elif self.model == "text-embedding-ada-002":
            return (num_tokens / 1_000_000) * 0.10
        else:  # Default to text-embedding-3-small
            return (num_tokens / 1_000_000) * 0.02
