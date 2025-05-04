import PyPDF2
from .base_loader import BaseDocumentLoader

class PDFLoader(BaseDocumentLoader):
    def load_document(self, file_path):
        import pypdf
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text