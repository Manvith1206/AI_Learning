from .base_chunker import BaseChunker
import re

class PageChunker(BaseChunker):
    
    def split_text(self, text):
        if not text:
            return []
        # Split the text into pages based on the number of pages
        pattern = r'--- Page (\d+):.*?---\n(.*?)(?=\n--- Page \d+:|$)'  # \Z means end of string
        pages = re.split("--- Page", text)
        
        
        chunks = []
        for page in pages:
            # Clean up the page text
            page = page.strip()
            if page:
                chunks.append(page)
                print("Page: ", page)

        
        return chunks