import docx2txt
from .base_loader import BaseDocumentLoader

class DOCXLoader(BaseDocumentLoader):
    def load_document(self, file_path):
        return docx2txt.process(file_path)
