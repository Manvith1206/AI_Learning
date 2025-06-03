from abc import ABC, abstractmethod
import csv
import docx2txt
import pypdf
import os

class BaseDocumentLoader(ABC):
    @abstractmethod
    def load_document(self, file_path: str) -> str:
        pass

class CSVLoader(BaseDocumentLoader):
    def load_document(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text += " ".join(row) + "\n"
        return text

class DOCXLoader(BaseDocumentLoader):
    def load_document(self, file_path: str) -> str:
        return docx2txt.process(file_path)

class PDFLoader(BaseDocumentLoader):
    def load_document(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            raise ValueError(f"Could not extract text from PDF: {file_path}") from e
        return text

class TXTLoader(BaseDocumentLoader):
    def load_document(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def get_loader(file_path: str, file_type: str = None) -> BaseDocumentLoader:
    if file_type is None:
        file_type = os.path.splitext(file_path)[1].lower()
    if file_type == ".pdf": return PDFLoader()
    elif file_type == ".docx": return DOCXLoader()
    elif file_type == ".txt": return TXTLoader()
    elif file_type == ".csv": return CSVLoader()
    else: raise ValueError(f"Unsupported file type: {file_type} for file: {file_path}")
