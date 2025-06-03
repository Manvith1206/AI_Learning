from .base_loader import BaseDocumentLoader

class TXTLoader(BaseDocumentLoader):
    def load_document(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
