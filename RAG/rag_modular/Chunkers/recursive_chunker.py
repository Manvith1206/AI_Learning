from langchain.text_splitter import TokenTextSplitter
from .base_chunker import BaseChunker
import tiktoken
from langchain_text_splitters.base import Tokenizer


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size=600, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # Define a length function that counts tokens
    def token_length_function(self, text: str):
        return len(self.encoding.encode(text))

    def split_text(self, text):
        if not text:
            return []
        
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,  # Max tokens per chunk
            chunk_overlap=self.chunk_overlap,  # Tokens to overlap
            encoding_name="cl100k_base"  # Tokenizer for OpenAI models
        )
        
        chunks = text_splitter.split_text(text=text)
        

        return chunks
