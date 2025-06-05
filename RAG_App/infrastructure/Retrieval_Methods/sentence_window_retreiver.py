import re
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import infrastructure.Common.RAG_Constants as constants
from .base_retriever import BaseRetriever

class SentenceWindowRetriever(BaseRetriever):
    """
    A retrieval enhancement layer that applies sentence window retrieval
    on top of an existing retrieval system.
    
    This assumes you already have:
    1. Documents chunked and stored
    2. A way to retrieve relevant chunks based on a query
    3. A way to get embeddings for sentences
    """
    
    def __init__(self, window_size: int = 2):
        """
        Initialize the sentence window retriever
        
        Args:
            window_size: Number of sentences to include on each side of the matched sentence
        """
        self.window_size = window_size
        # Regex pattern for sentence tokenization
        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex
        
        Args:
            text: The text to split
            
        Returns:
            List of sentences
        """
        
        # Split text using regex
        sentences = self.sentence_pattern.split(text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            # Remove leading/trailing whitespace
            sentence = sentence.strip()
            if sentence:  # Skip empty sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def apply_sentence_window_retrieval(
        self,
        query: str,
        retrieved_chunks: List[str],
        embedding_function=None,  # Optional function to get embeddings
        **func_kwargs
    ) -> List[str]:
        """
        Enhance retrieved chunks by applying sentence window retrieval
        
        Args:
            query: The user's query
            retrieved_chunks: List of text chunks retrieved by your existing system
            embedding_function: Optional function that takes text and returns embeddings
                                If None, will use basic string matching for demonstration
        
        Returns:
            List of enhanced context windows
        """
        # Step 1: Split all chunks into sentences
        all_sentences = []
        chunk_to_sentences = {}
        sentence_to_chunk = {}
        sentence_indices = {}  # Track the position of each sentence within its original chunk
        
        for chunk_idx, chunk in enumerate(retrieved_chunks):
            sentences = self.split_into_sentences(chunk)
            chunk_to_sentences[chunk_idx] = sentences
            
            for sent_idx, sentence in enumerate(sentences):
                sentence_idx = len(all_sentences)
                all_sentences.append(sentence)
                sentence_to_chunk[sentence_idx] = chunk_idx
                sentence_indices[sentence_idx] = sent_idx
        
        # Step 2: Find the most relevant sentences for the query
        if embedding_function:
            
            # Use provided embedding of query if available
            query_emb = func_kwargs.pop('query_embedding', None)
            # Sentence embeddings
            sentence_embeddings = embedding_function(all_sentences, **func_kwargs)
            # Query embedding: use provided or compute
            if query_emb is None:
                query_emb = embedding_function([query], **func_kwargs)[0]
            # else assume it's already a flat vector
            similarities = cosine_similarity([query_emb], sentence_embeddings)[0]
            
            # Get top sentence indices
            top_sentence_indices = np.argsort(similarities)[-5:][::-1]  # Get top 5 most similar sentences
        else:
            
            # Simple fallback using basic word overlap if no embedding function provided
            word_overlaps = []
            query_words = set(query.lower().split())
            
            for sentence in all_sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                word_overlaps.append(overlap)
            
            top_sentence_indices = np.argsort(word_overlaps)[-5:][::-1]  # Get top 5 sentences with most word overlap
        
        # Step 3: For each top sentence, get its window of surrounding sentences
        enhanced_contexts = []
        
        for sentence_idx in top_sentence_indices:
            chunk_idx = sentence_to_chunk[sentence_idx]
            sent_idx_in_chunk = sentence_indices[sentence_idx]
            
            # Get sentences in window (respecting chunk boundaries)
            sentences_in_chunk = chunk_to_sentences[chunk_idx]
            start_idx = max(0, sent_idx_in_chunk - self.window_size)
            end_idx = min(len(sentences_in_chunk) - 1, sent_idx_in_chunk + self.window_size)
            
            window_sentences = sentences_in_chunk[start_idx:end_idx + 1]
            window_text = " ".join(window_sentences)
            
            enhanced_contexts.append(window_text)
        
        # Step 4: Merge overlapping windows
        merged_contexts = self.merge_overlapping_contexts(enhanced_contexts)
        
        return merged_contexts
    
    def merge_overlapping_contexts(self, contexts: List[str]) -> List[str]:
        """Merge contexts that have significant overlap"""
        if not contexts:
            return []
        
        # Convert contexts to sets of sentences for easier overlap detection
        context_sentences = []
        for context in contexts:
            sentences = set(self.split_into_sentences(context))
            context_sentences.append(sentences)
        
        # Merge overlapping contexts
        merged = []
        current = context_sentences[0]
        current_text = contexts[0]
        
        for i in range(1, len(context_sentences)):
            if len(current.intersection(context_sentences[i])) > 0:
                # Contexts overlap, merge them
                # For merging text, we'll use a simple approach - just concatenate and split again
                combined_text = current_text + " " + contexts[i]
                combined_sentences = self.split_into_sentences(combined_text)
                # Remove duplicates while preserving order
                seen = set()
                unique_sentences = []
                for s in combined_sentences:
                    if s not in seen:
                        seen.add(s)
                        unique_sentences.append(s)
                
                current_text = " ".join(unique_sentences)
                current = set(unique_sentences)
            else:
                # No overlap, add current to results and start a new one
                merged.append(current_text)
                current = context_sentences[i]
                current_text = contexts[i]
        
        # Add the last context
        merged.append(current_text)
        
        return merged

    def retrieve(self, query_embedding, documents, **kwargs):
        
        query_text = kwargs.get(constants.QUERY_TEXT)
        
        if not query_text:
            raise ValueError(constants.QUERY_TEXT_MUST_BE_PROVIDED_ERROR_MESSAGE)
        
        # Apply sentence window retrieval
        retriever = SentenceWindowRetriever(window_size=self.window_size)
        enhanced_contexts = retriever.apply_sentence_window_retrieval(
            query_text,
            documents,
            embedding_function=None,
            **kwargs
        )
        return enhanced_contexts

# Example usage with a simple embedding function for demonstration
def simple_embedding_function(texts,  **kwargs):
    """
    A placeholder for your actual embedding function.
    In a real system, this would use your preferred embedding method.
    """
    
    vector_store = kwargs.get(constants.CONFIG_EMBEDDER)
    if not vector_store:
        raise ValueError(constants.VECTOR_STORE_MUST_BE_PROVIDED_ERROR_MESSAGE)

    embeddings = vector_store.fit(texts)
    
    return embeddings