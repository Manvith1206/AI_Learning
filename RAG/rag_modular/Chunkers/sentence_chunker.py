from .base_chunker import BaseChunker
import re

class SentenceChunker(BaseChunker):
    def __init__(self, max_sentences=5):
        self.max_sentences = max_sentences

    def split_text(self, text):
        if not text:
            return []
        # Use simple regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), self.max_sentences):
            chunk = ' '.join(sentences[i:i+self.max_sentences])
            chunks.append(chunk)
        return chunks