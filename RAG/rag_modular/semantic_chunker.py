import numpy as np
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Any
from base_chunker import BaseChunker
import rag_modular.RAG_Constants as constants
max_sentences = 300

class SemanticChunker(BaseChunker):

    def __init__(self, 
                 model_name: str = constants.SENTENCE_TRANSFORMER_MODEL_ALL_MINI, 
                 similarity_threshold: float = 0.7,
                 min_chunk_size: int = 3,
                 max_chunk_size: int = 20):
        """
        Initialize the semantic chunker.
        
        Args:
            model_name: Name of the sentence transformer model to use
            similarity_threshold: Threshold below which a semantic boundary is identified
            min_chunk_size: Minimum number of sentences in a chunk
            max_chunk_size: Maximum number of sentences in a chunk
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    def split_text(self, text):
        # Chunk the document
        chunks = self.chunk_text(text)

        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _segment_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex pattern matching.
        This is an alternative to NLTK's sentence tokenization.
        """
        preprocessed_text = self._preprocess_text(text)
        
        # Split on sentence-ending punctuation followed by whitespace and (optional) quotation marks
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"])'
        sentences = re.split(sentence_pattern, preprocessed_text)
        
        # Further split any remaining sentences that might be too long
        final_sentences = []
        for sentence in sentences:
            if len(sentence) > max_sentences:  # If a sentence is too long, split on commas or semicolons
                split_parts = re.split(r'(?<=[,;])\s+', sentence)
                if len(split_parts) > 1:
                    final_sentences.extend(split_parts)
                else:
                    final_sentences.append(sentence)
            else:
                final_sentences.append(sentence)
        
        # Filter out empty sentences
        return [s for s in final_sentences if s.strip()]
    
    def _find_chunk_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """
        Find semantic boundaries based on similarity between adjacent sentence embeddings.
        Returns list of indices where chunks should end.
        """
        boundaries = []
        current_chunk_size = 0
        
        for i in range(len(embeddings) - 1):
            current_chunk_size += 1
            similarity = float(self._cosine_similarity(embeddings[i], embeddings[i+1]))
            
            # Create boundary if similarity is below threshold or max chunk size is reached
            if (similarity < self.similarity_threshold and current_chunk_size >= self.min_chunk_size) or \
               current_chunk_size >= self.max_chunk_size:
                boundaries.append(i + 1)  # End chunk at the next sentence
                current_chunk_size = 0
        
        # Add the last boundary if needed
        if boundaries and boundaries[-1] != len(embeddings):
            boundaries.append(len(embeddings))
        elif not boundaries:
            boundaries.append(len(embeddings))
        
        
        return boundaries
    
    def chunk_text(self, text: str) -> List[Dict[Any, Any]]:
        """
        Main method to semantically chunk the input text.
        
        Args:
            text: The document text to chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Split text into sentences
        
        sentences = self._segment_into_sentences(text)
        if not sentences:
            return []
            
        # Calculate embeddings for each sentence
        embeddings = self.model.encode(sentences)
        
        # Find chunk boundaries
        boundaries = self._find_chunk_boundaries(embeddings)
        
        # Create chunks based on boundaries
        chunks = []
        start_idx = 0
        for end_idx in boundaries:
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            chunks.append(chunk_text)
            start_idx = end_idx
            
        return chunks
