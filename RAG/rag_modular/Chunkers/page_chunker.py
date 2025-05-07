from .base_chunker import BaseChunker

class PageChunker(BaseChunker):
    def __init__(self):
        
    def split_text(self, text):
        if not text:
            return []
        # Split the text into pages based on the number of pages
        pages = text.split('\f')  # Assuming '\f' is the page delimiter
        chunks = []
        for i in range(0, len(pages), self.max_pages):
            chunk = '\f'.join(pages[i:i+self.max_pages])
            chunks.append(chunk)
        return chunks