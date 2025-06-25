import time
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker
import infrastructure.common.rag_constants as constants
from infrastructure.common.component_registry import CHUNKERS_REGISTRY

@CHUNKERS_REGISTRY.register(constants.ChunkerType.RECURSIVE.value)
class RecursiveChunker(BaseChunker):
    """
    A chunker that uses a recursive character text splitter from LangChain.
    It splits text based on a hierarchy of separators and measures chunk size by token count.
    """
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 200):
        """
        Initializes the RecursiveChunker.

        Args:
            chunk_size (int): The maximum size of each chunk in tokens.
            chunk_overlap (int): The number of tokens to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._time_taken = 0.0
        self._cost = 0.0

    def split_text(self, text: str) -> List[str]:
        """
        Splits the text using a token-based recursive character splitter.

        Args:
            text (str): The text to be split.

        Returns:
            List[str]: A list of text chunks.
        """
        start_time = time.time()
        if not text:
            self._time_taken = time.time() - start_time
            return []

        # Uses a text splitter that is aware of token counts
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=constants.ENCODING_NAME_FOR_TOKEN_COUNT,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        chunks = text_splitter.split_text(text)
        
        self._time_taken = time.time() - start_time
        return chunks

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the time taken and cost for the last split operation.
        For this chunker, the cost is always zero.

        Returns:
            Tuple[float, float]: A tuple containing the cost (0.0) and the time taken in seconds.
        """
        return self._cost, self._time_taken