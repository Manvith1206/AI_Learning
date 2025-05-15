import csv
from .base_loader import BaseDocumentLoader

class CSVLoader(BaseDocumentLoader):
    def load_document(self, file_path):
        text = ""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text += " ".join(row) + "\n"
        return text
