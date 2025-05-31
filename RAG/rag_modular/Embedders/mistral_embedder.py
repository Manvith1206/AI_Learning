from mistralai import Mistral
from .base_embedder import BaseEmbedder
import streamlit as st
import rag_modular.Common.RAG_Constants as constants
import tiktoken
import random, time, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class MistralEmbedder(BaseEmbedder):
    """Mistral embedding model implementation with batching and rate limiting"""
    def __init__(self, api_key, model="mistral-embed", batch_size=20, 
                 initial_delay=1, max_retries=5, max_delay=60):
        """Initialize the Mistral embedding model"""
        api_key = api_key
        if not api_key:
            raise ValueError("Mistral API key not found in environment variables")
        
        self._client = Mistral(api_key=api_key)
        self._model_name = model
        self._batch_size = batch_size
        self._initial_delay = initial_delay
        self._max_retries = max_retries
        self._max_delay = max_delay
        self.time_taken = 0
        self.cost = 0
        
        # Initialize tokenizer for token counting
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self._tokenizer.encode(text))
    
    def _batch_texts(self, texts, max_tokens_per_batch: int = 7000):
        """Split texts into batches based on token count"""
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for text in texts:
            text_tokens = self._count_tokens(text)
            # If this single text is too large for a batch, we need special handling
            if text_tokens > max_tokens_per_batch:
                # If we have anything in the current batch, add it to batches
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0
                
                # This is a big text that needs special handling
                # For simplicity, we'll just put it in its own batch
                batches.append([text])
                continue
            
            # If adding this text would exceed batch token limit, create a new batch
            if current_batch_tokens + text_tokens > max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = [text]
                current_batch_tokens = text_tokens
            else:
                current_batch.append(text)
                current_batch_tokens += text_tokens
        
        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _call_api_with_backoff(self, inputs):
        """Call the Mistral API with exponential backoff for rate limiting"""
        retry_count = 0
        delay = self._initial_delay
        
        while True:
            try:
                response = self._client.embeddings.create(
                    model=self._model_name,
                    inputs=inputs
                )
                # If we get here, the request was successful
                return response
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_message and "rate limit" in error_message.lower():
                    retry_count += 1
                    
                    if retry_count > self._max_retries:
                        logger.error(f"Maximum retries ({self._max_retries}) exceeded. Giving up.")
                        raise e
                    
                    # Add jitter to avoid all clients retrying at the same time
                    jitter = random.uniform(0, 0.1 * delay)
                    actual_delay = min(delay + jitter, self._max_delay)
                    
                    logger.warning(f"Rate limit exceeded. Retrying in {actual_delay:.2f} seconds (retry {retry_count}/{self._max_retries})")
                    time.sleep(actual_delay)
                    
                    # Exponential backoff
                    delay = min(delay * 2, self._max_delay)
                else:
                    # If it's not a rate limit error, re-raise
                    raise e
    
    def transform(self, texts):
        """Convert a query string to an embedding vector with retry logic"""
        start_time = time.time()
        # Handle single text or list of texts
        if isinstance(texts, str):
            texts = [texts]
            
        response = self._call_api_with_backoff(texts)
        
        # Extract embeddings from response
        embeddings = []
        for item in response.data:
            embeddings.append(item.embedding)
        end_time = time.time()

        self.time_taken = end_time - start_time
        # Return a list of embeddings (2D array)
        return embeddings
    
    def fit(self, texts ):
        """Convert a list of document strings to embedding vectors with automatic batching and rate limiting"""
        start_time = time.time()
        all_embeddings = []
        
        # Create batches based on token count
        batches = self._batch_texts(texts)
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            
            # Add delay between batches to avoid rate limiting
            if i > 0:
                time.sleep(1)  # 1 second delay between batches
            
            response = self._call_api_with_backoff(batch)
            
            # Extract embeddings from the response and add to results
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        end_time = time.time()
        self.time_taken = end_time - start_time
        return all_embeddings
    
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        return 1024  # Dimension for Mistral embeddings