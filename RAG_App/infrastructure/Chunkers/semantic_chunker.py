import time
import numpy as np
import re
from typing import List, Tuple
from .base_chunker import BaseChunker
from ..embedders.base_embedder import BaseEmbedder
from ..common.component_registry import CHUNKERS_REGISTRY
import infrastructure.common.RAG_Constants as constants

@CHUNKERS_REGISTRY.register(constants.ChunkerType.SEMANTIC.value)
class SemanticChunker(BaseChunker):
    """
    A chunker that splits text based on semantic similarity between sentences.
    It uses a provided embedder to generate sentence embeddings and then identifies
    split points where the similarity drops below a certain threshold.
    """

    def __init__(self,
                 embedder: BaseEmbedder,
                 similarity_threshold: float = 0.7,
                 min_chunk_size: int = 3,
                 max_chunk_size: int = 20,
                 max_sentence_length: int = 300):
        """
        Initializes the semantic chunker.

        Args:
            embedder (BaseEmbedder): An instance of a class that inherits from BaseEmbedder.
            similarity_threshold (float): Threshold below which a semantic boundary is identified.
            min_chunk_size (int): Minimum number of sentences in a chunk.
            max_chunk_size (int): Maximum number of sentences in a chunk.
            max_sentence_length (int): Maximum character length for a sentence before it's split.
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_sentence_length = max_sentence_length
        self._time_taken = 0.0
        self._cost = 0.0

    def split_text(self, text: str) -> List[str]:
        """
        Main entry point for splitting text, consistent with the BaseChunker interface.
        """
        start_time = time.time()
        if not text:
            self._time_taken = time.time() - start_time
            self._cost = 0.0
            return []

        chunks = self._chunk_text(text)

        embedder_cost, _ = self.embedder.get_cost_and_time_taken()
        self._cost = embedder_cost
        self._time_taken = time.time() - start_time
        return chunks

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the time taken and cost for the last split operation.
        """
        return self._cost, self._time_taken

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _preprocess_text(self, text: str) -> str:
        """Performs basic text preprocessing."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _segment_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences using regex pattern matching.
        """
        preprocessed_text = self._preprocess_text(text)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"])'
        sentences = re.split(sentence_pattern, preprocessed_text)

        final_sentences = []
        for sentence in sentences:
            if len(sentence) > self.max_sentence_length:
                split_parts = re.split(r'(?<=[,;])\s+', sentence)
                final_sentences.extend(split_parts)
            else:
                final_sentences.append(sentence)

        return [s for s in final_sentences if s.strip()]

    def _find_chunk_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """
        Finds semantic boundaries based on similarity between adjacent sentence embeddings.
        """
        boundaries = []
        current_chunk_size = 0

        if len(embeddings) <= 1:
            return [len(embeddings)]

        for i in range(len(embeddings) - 1):
            current_chunk_size += 1
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            if (similarity < self.similarity_threshold and current_chunk_size >= self.min_chunk_size) or \
               current_chunk_size >= self.max_chunk_size:
                boundaries.append(i + 1)
                current_chunk_size = 0

        if not boundaries or boundaries[-1] != len(embeddings):
            boundaries.append(len(embeddings))

        return boundaries

    def _chunk_text(self, text: str) -> List[str]:
        """
        Performs the main logic to semantically chunk the input text.
        """
        sentences = self._segment_into_sentences(text)
        if not sentences:
            return []

        embeddings_result = self.embedder.embed_documents(sentences)
        
        if embeddings_result and isinstance(embeddings_result[0], np.ndarray):
            embeddings = embeddings_result
        elif embeddings_result and hasattr(embeddings_result[0], 'values'):
            embeddings = [e.values for e in embeddings_result]
        elif embeddings_result and hasattr(embeddings_result[0], 'embedding'):
            embeddings = [e.embedding for e in embeddings_result]
        else:
            embeddings = []

        if not embeddings:
            return [' '.join(sentences)]

        boundaries = self._find_chunk_boundaries(np.array(embeddings))

        chunks = []
        start_idx = 0
        for end_idx in boundaries:
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            chunks.append(chunk_text)
            start_idx = end_idx

        return chunks
