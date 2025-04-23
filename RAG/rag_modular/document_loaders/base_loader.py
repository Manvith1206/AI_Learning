from abc import ABC, abstractmethod

class BaseDocumentLoader(ABC):
    @abstractmethod
    def load_document(self, file_path):
        pass
