import time
import numpy as np
import re
from infrastructure.chunkers.base_chunker import BaseChunker
from infrastructure.embedders.base_embedder import BaseEmbedder
from infrastructure.common.component_registry import register, CHUNKERS_REGISTRY
import infrastructure.common.rag_constants as constants

max_sentences = 300

@register(CHUNKERS_REGISTRY, constants.ChunkerType.SEMANTIC.value)
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
                 max_chunk_size: int = 20):
        """
        Initialize the semantic chunker.
        
        Args:
            embedder: An instance of a class that inherits from BaseEmbedder.
            similarity_threshold: Threshold below which a semantic boundary is identified.
            min_chunk_size: Minimum number of sentences in a chunk.
            max_chunk_size: Maximum number of sentences in a chunk.
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.time_taken = 0
        self.cost = 0

    def split_text(self, text):
        """
        Main entry point for splitting text, consistent with BaseChunker interface.
        """
        start_time = time.time()
        chunks = self._chunk_text(text)
        end_time = time.time()
        self.time_taken = end_time - start_time
        return chunks
    
    def get_cost_and_time_taken(self):
        """
        Returns the time taken and cost for the last split operation.
        """
        return self.cost, self.time_taken

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _segment_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex pattern matching.
        """
        preprocessed_text = self._preprocess_text(text)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"])'
        sentences = re.split(sentence_pattern, preprocessed_text)
        
        final_sentences = []
        for sentence in sentences:
            if len(sentence) > max_sentences:
                split_parts = re.split(r'(?<=[,;])\s+', sentence)
                final_sentences.extend(split_parts)
            else:
                final_sentences.append(sentence)
        
        return [s for s in final_sentences if s.strip()]
    
    def _find_chunk_boundaries(self, embeddings: np.ndarray) -> list[int]:
        """
        Find semantic boundaries based on similarity between adjacent sentence embeddings.
        """
        boundaries = []
        current_chunk_size = 0
        
        if len(embeddings) <= 1:
            return [len(embeddings)]

        for i in range(len(embeddings) - 1):
            current_chunk_size += 1
            similarity = self._cosine_similarity(embeddings[i], embeddings[i+1])
            
            if (similarity < self.similarity_threshold and current_chunk_size >= self.min_chunk_size) or \
               current_chunk_size >= self.max_chunk_size:
                boundaries.append(i + 1)
                current_chunk_size = 0
        
        if not boundaries or boundaries[-1] != len(embeddings):
            boundaries.append(len(embeddings))
        
        return boundaries
    
    def _chunk_text(self, text: str) -> list[str]:
        """
        Main method to semantically chunk the input text.
        """
        sentences = self._segment_into_sentences(text)
        if not sentences:
            return []
            
        # The embedder is expected to return a list of embeddings (np.ndarray)
        embeddings = self.embedder.embed_documents(sentences)
        
        boundaries = self._find_chunk_boundaries(np.array(embeddings))
        
        chunks = []
        start_idx = 0
        for end_idx in boundaries:
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            chunks.append(chunk_text)
            start_idx = end_idx
            
        return chunks
