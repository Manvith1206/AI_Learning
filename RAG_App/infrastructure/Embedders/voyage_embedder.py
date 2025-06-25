import time
import logging
from typing import List, Tuple
import voyageai
from .base_embedder import BaseEmbedder
from ..common.component_registry import EMBEDDERS_REGISTRY
import infrastructure.common.RAG_Constants as constants

logger = logging.getLogger(__name__)

@EMBEDDERS_REGISTRY.register(constants.EmbedderType.VOYAGE.value)
class VoyageEmbedder(BaseEmbedder):
    """An embedder that uses the Voyage AI API to generate text embeddings."""

    def __init__(self, api_key: str, model: str = constants.VoyageEmbedModels.VOYAGE_EMBED_DEFAULT_MODEL.value, batch_size: int = 128):
        """
        Initializes the VoyageEmbedder.

        Args:
            api_key (str): The Voyage AI API key.
            model (str): The Voyage embedding model to use.
            batch_size (int): The number of documents to process in a single batch.
        """
        if not api_key:
            raise ValueError("Voyage AI API key is required.")
        
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        # Voyage has a batch size limit of 128
        self.batch_size = min(batch_size, 128)
        self._time_taken = 0.0
        self._cost = 0.0

    def _batch_chunks(self, texts: List[str]) -> List[List[str]]:
        """Yields successive batches of a specified size."""
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents using the Voyage AI API.

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
                response = self.client.embed(texts=batch, model=self.model, input_type="document")
                all_embeddings.extend(response.embeddings)
                total_tokens += response.total_tokens
            except Exception as e:
                logger.error(f"Error embedding batch with Voyage AI: {e}")
                all_embeddings.extend([[]] * len(batch))

        self._cost = self._get_cost(total_tokens)
        self._time_taken = time.time() - start_time
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query using the Voyage AI API.

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
            response = self.client.embed(texts=[query], model=self.model, input_type="query")
            embedding = response.embeddings[0]
            self._cost = self._get_cost(response.total_tokens)
        except Exception as e:
            logger.error(f"Error embedding query with Voyage AI: {e}")
            embedding = []
            self._cost = 0.0

        self._time_taken = time.time() - start_time
        return embedding

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the cost and time taken for the last embedding operation.
        """
        return self._cost, self._time_taken

    def _get_cost(self, tokens: int) -> float:
        """
        Calculates the cost based on the number of tokens for the specific model.
        Pricing is per 1,000,000 tokens.
        - voyage-2, voyage-large-2: $0.10
        - voyage-code-2: $0.10
        """
        # As of late 2023, most common models are priced similarly.
        # Using a single rate for simplicity.
        return (tokens / 1_000_000) * 0.10