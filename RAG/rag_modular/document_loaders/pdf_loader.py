import PyPDF2
from .base_loader import BaseDocumentLoader

class PDFLoader(BaseDocumentLoader):
    def load_document(self, file_path):
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
        return text
