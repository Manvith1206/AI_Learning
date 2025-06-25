import time
import logging
from typing import List, Tuple
import google.generativeai as genai
from .base_embedder import BaseEmbedder
from ..common.component_registry import EMBEDDERS_REGISTRY
import infrastructure.common.rag_constants as constants

logger = logging.getLogger(__name__)

@EMBEDDERS_REGISTRY.register(constants.EmbedderType.GEMINI.value)
class GeminiEmbedder(BaseEmbedder):
    """
    An embedder that uses the Google Gemini API to generate text embeddings.
    """
    def __init__(self, api_key: str, model_name: str = constants.GeminiEmbedModels.GEMINI_EMBED_001_MODEL.value):
        """
        Initializes the GeminiEmbedder.

        Args:
            api_key (str): The Google API key.
            model_name (str): The Gemini embedding model to use.
        """
        if not api_key:
            raise ValueError("Google API key is required.")
        
        genai.configure(api_key=api_key)
        self.model = model_name
        self._cost = 0.0
        self._time_taken = 0.0
        # As per Google's documentation, the batch size limit is 100.
        self.batch_size = 100

    def _batch_chunks(self, texts: List[str]) -> List[List[str]]:
        """Yields successive batches of a specified size."""
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents using the Gemini API.

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
        total_chars = 0

        for batch in self._batch_chunks(texts):
            try:
                response = genai.embed_content(
                    model=self.model,
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                all_embeddings.extend(response['embedding'])
                total_chars += sum(len(text) for text in batch)
            except Exception as e:
                logger.error(f"Error embedding batch with Gemini: {e}")
                all_embeddings.extend([[]] * len(batch))

        self._cost = self._get_cost(total_chars)
        self._time_taken = time.time() - start_time
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query using the Gemini API.

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
            response = genai.embed_content(
                model=self.model,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            embedding = response['embedding']
            self._cost = self._get_cost(len(query))
        except Exception as e:
            logger.error(f"Error embedding query with Gemini: {e}")
            embedding = []
            self._cost = 0.0
        
        self._time_taken = time.time() - start_time
        return embedding

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the cost and time taken for the last embedding operation.
        """
        return self._cost, self._time_taken
    
    def _get_cost(self, num_chars: int) -> float:
        """
        Calculates the cost based on the number of characters.
        Gemini embedding models are currently free of charge. This is a placeholder.
        Pricing: $0.10 / 1M characters
        """
        return (num_chars / 1_000_000) * 0.10