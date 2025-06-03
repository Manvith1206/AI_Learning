from langchain.text_splitter import TokenTextSplitter
from .base_chunker import BaseChunker
import tiktoken
from langchain_text_splitters.base import Tokenizer

import time

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size=600, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.time_taken = 0
        self.cost = 0

    # Define a length function that counts tokens
    def token_length_function(self, text: str):
        return len(self.encoding.encode(text))

    def split_text(self, text):
        start_time = time.time()
        if not text:
            return []
        
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,  # Max tokens per chunk
            chunk_overlap=self.chunk_overlap,  # Tokens to overlap
            encoding_name="cl100k_base"  # Tokenizer for OpenAI models
        )
        
        chunks = text_splitter.split_text(text=text)
        
        end_time = time.time()
        self.time_taken = end_time - start_time
        return chunks

    def get_cost_and_time_taken(self):
        """
        Returns the time taken and cost for the last split operation.
        """
        return self.cost, self.time_taken