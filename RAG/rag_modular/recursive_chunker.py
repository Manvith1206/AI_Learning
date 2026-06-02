from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker
class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size=600, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        if not text:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)
