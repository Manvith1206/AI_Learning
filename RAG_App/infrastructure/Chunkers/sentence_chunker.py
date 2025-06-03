import time
from .base_chunker import BaseChunker
import re

class SentenceChunker(BaseChunker):
    def __init__(self, max_sentences=5):
        self.max_sentences = max_sentences
        self.time_taken = 0
        self.cost = 0

    def split_text(self, text):
        start_time = time.time()
        if not text:
            return []
        # Use simple regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), self.max_sentences):
            chunk = ' '.join(sentences[i:i+self.max_sentences])
            chunks.append(chunk)

        end_time = time.time()
        self.time_taken = end_time - start_time
        return chunks
    def get_cost_and_time_taken(self):
        """
        Returns the time taken and cost for the last split operation.
        """
        # Since this chunker does not involve any external API calls, we can return 0 for both time and cost
        return self.cost, self.time_taken