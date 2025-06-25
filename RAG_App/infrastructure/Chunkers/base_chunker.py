from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseChunker(ABC):
    """Abstract base class for text chunking components."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Splits the input text into a list of smaller chunks.

        Args:
            text (str): The text to be split.

        Returns:
            List[str]: A list of text chunks.
        """
        pass

    @abstractmethod
    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the time taken and estimated cost for the last split operation.

        Returns:
            Tuple[float, float]: A tuple containing the cost (e.g., in USD) and the time taken (in seconds).
        """
        pass