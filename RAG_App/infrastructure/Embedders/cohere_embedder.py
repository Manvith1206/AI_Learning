import time
import cohere
import logging
from typing import List, Tuple
from .base_embedder import BaseEmbedder
from ..common.component_registry import EMBEDDERS_REGISTRY
import infrastructure.common.RAG_Constants as constants

logger = logging.getLogger(__name__)

@EMBEDDERS_REGISTRY.register(constants.EmbedderType.COHERE.value)
class CohereEmbedder(BaseEmbedder):
    """
    An embedder that uses the Cohere API to generate text embeddings.
    """
    def __init__(self, api_key: str, model: str = constants.CohereEmbedModels.COHERE_EMBED_MODEL_V3_ENG.value):
        """
        Initializes the CohereEmbedder.

        Args:
            api_key (str): The Cohere API key.
            model (str): The Cohere embedding model to use.
        """
        if not api_key:
            raise ValueError("Cohere API key is required.")
        self.client = cohere.Client(api_key)
        self.model = model
        self._cost = 0.0
        self._time_taken = 0.0

    def _batch_chunks(self, chunks: List[str], batch_size: int = 96) -> List[List[str]]:
        """Yields successive batches of a specified size."""
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents using the Cohere API.

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
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document"
            )
            if response.meta and response.meta.billed_units and response.meta.billed_units.input_tokens:
                total_tokens += response.meta.billed_units.input_tokens
            else:
                logger.warning("Cohere API response did not include input_tokens. Cost metric might be inaccurate.")
            
            all_embeddings.extend(response.embeddings)

        self._cost = self._get_cost_based_on_model(total_tokens)
        self._time_taken = time.time() - start_time
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query using the Cohere API.

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

        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="search_query"
        )
        
        total_tokens = 0
        if response.meta and response.meta.billed_units and response.meta.billed_units.input_tokens:
            total_tokens = response.meta.billed_units.input_tokens
        else:
            logger.warning("Cohere API response did not include input_tokens. Cost metric might be inaccurate.")

        self._cost = self._get_cost_based_on_model(total_tokens)
        self._time_taken = time.time() - start_time
        return response.embeddings[0]

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the cost and time taken for the last embedding operation.
        """
        return self._cost, self._time_taken

    def _get_cost_based_on_model(self, tokens: int) -> float:
        """
        Calculates the cost based on the model and number of tokens.
        Prices are based on official Cohere documentation.
        """
        if self.model == constants.CohereEmbedModels.COHERE_EMBED_MODEL_V3_ENG.value:
            return (tokens / 1_000_000) * 0.10  # $0.10 per 1M tokens for v3
        elif self.model == constants.CohereEmbedModels.COHERE_EMBED_MODEL_V2_ENG.value:
            return (tokens / 1_000_000) * 0.10  # $0.10 per 1M tokens for v2
        else:
            logger.warning(f"Cost calculation not implemented for model '{self.model}'. Defaulting to 0.")
            return 0.0