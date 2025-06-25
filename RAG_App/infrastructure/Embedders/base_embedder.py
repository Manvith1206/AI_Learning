from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedder components.
    Defines the interface for embedding documents and queries.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): A list of documents to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query.

        Args:
            query (str): The query string to embed.

        Returns:
            List[float]: The embedding for the query.
        """
        pass

    @abstractmethod
    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the total cost and time taken for the embedding operations.

        Returns:
            Tuple[float, float]: A tuple containing the cost and the time taken in seconds.
        """
        pass