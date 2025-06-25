import re
import time
from typing import List, Tuple
from .base_chunker import BaseChunker
from infrastructure.common.component_registry import CHUNKERS_REGISTRY
import infrastructure.common.rag_constants as constants

@CHUNKERS_REGISTRY.register(constants.ChunkerType.PAGE.value)
class PageChunker(BaseChunker):
    """
    A chunker that splits text based on page delimiters.
    It assumes that pages are separated by a '--- Page' marker.
    """
    def __init__(self):
        self._time_taken = 0.0
        self._cost = 0.0

    def split_text(self, text: str) -> List[str]:
        """
        Splits the text by page delimiters.

        Args:
            text (str): The input text containing page markers.

        Returns:
            List[str]: A list of strings, where each string is the content of a page.
        """
        start_time = time.time()
        if not text:
            self._time_taken = time.time() - start_time
            return []

        # Split by '--- Page X ---' and filter out any empty strings resulting from the split.
        pages = re.split(r'--- Page \d+ ---', text)
        chunks = [page.strip() for page in pages if page and page.strip()]
        
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