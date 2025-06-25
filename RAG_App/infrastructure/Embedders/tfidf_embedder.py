import time
import logging
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from .base_embedder import BaseEmbedder
from ..common.component_registry import EMBEDDERS_REGISTRY
import infrastructure.common.RAG_Constants as constants

logger = logging.getLogger(__name__)

@EMBEDDERS_REGISTRY.register(constants.EmbedderType.TFIDF.value)
class TFIDFEmbedder(BaseEmbedder):
    """A local, non-API-based embedder using TF-IDF vectorization."""

    def __init__(self):
        """Initializes the TFIDFEmbedder."""
        self.vectorizer = TfidfVectorizer()
        self._time_taken = 0.0
        self._cost = 0.0  # Cost is always zero for local embedders

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Fits the TF-IDF vectorizer to the documents and transforms them into embeddings.

        Args:
            texts (List[str]): A list of documents to fit and transform.

        Returns:
            List[List[float]]: A list of dense TF-IDF embeddings.
        """
        start_time = time.time()
        if not texts:
            self._time_taken = time.time() - start_time
            return []

        try:
            vectors = self.vectorizer.fit_transform(texts).toarray().tolist()
        except Exception as e:
            logger.error(f"Error fitting and transforming documents with TF-IDF: {e}")
            vectors = [[] for _ in texts]

        self._time_taken = time.time() - start_time
        return vectors

    def embed_query(self, query: str) -> List[float]:
        """
        Transforms a single query into an embedding using the fitted TF-IDF vectorizer.

        Args:
            query (str): The query string to embed.

        Returns:
            List[float]: The dense TF-IDF embedding for the query.
        
        Raises:
            NotFittedError: If the vectorizer has not been fitted yet.
        """
        start_time = time.time()
        if not query:
            self._time_taken = time.time() - start_time
            return []

        try:
            # The first element of the transform result is the embedding for the query
            vector = self.vectorizer.transform([query]).toarray().tolist()[0]
        except NotFittedError as e:
            logger.error("TF-IDF model is not fitted. Call embed_documents first.")
            raise NotFittedError("TF-IDF model is not fitted. Call embed_documents first.") from e
        except Exception as e:
            logger.error(f"Error transforming query with TF-IDF: {e}")
            vector = []

        self._time_taken = time.time() - start_time
        return vector

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the cost and time taken for the last embedding operation.
        Cost is always 0 for this local embedder.
        """
        return self._cost, self._time_taken