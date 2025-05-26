from abc import ABC, abstractmethod

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text):
        pass
    @abstractmethod
    def get_cost_and_time_taken(self):
        """
        Returns the time taken and cost for the last split operation.
        """
        pass