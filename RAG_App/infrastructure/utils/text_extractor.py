import os
import PyPDF2
import docx

class BaseTextExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_text(self):
        raise NotImplementedError

class TxtExtractor(BaseTextExtractor):
    def extract_text(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return [f.read()]

class PdfExtractor(BaseTextExtractor):
    def extract_text(self):
        with open(self.file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return [page.extract_text() for page in reader.pages]

class DocxExtractor(BaseTextExtractor):
    def extract_text(self):
        doc = docx.Document(self.file_path)
        return [p.text for p in doc.paragraphs if p.text]

class TextExtractorFactory:
    @staticmethod
    def get_extractor(file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.txt':
            return TxtExtractor(file_path)
        elif ext == '.pdf':
            return PdfExtractor(file_path)
        elif ext == '.docx':
            return DocxExtractor(file_path)
        else:
            return None
