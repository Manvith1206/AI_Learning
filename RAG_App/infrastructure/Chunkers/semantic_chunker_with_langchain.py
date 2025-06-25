import time
from langchain_experimental.text_splitter import SemanticChunker
from infrastructure.chunkers.base_chunker import BaseChunker
from infrastructure.common.RAG_Constants import ThresholdTypeForSemanticChunkinLangChain

class SemanticChunkerWithLangChain(BaseChunker):
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.semantic_chunker = SemanticChunker(self.embed_model, breakpoint_threshold_type=ThresholdTypeForSemanticChunkinLangChain.PERCENTILE)
        self.time_taken = 0
        self.cost = 0

    def split_text(self, text):
        star_time = time.time()
        chunks = self.semantic_chunker.split_text(text=text)
        end_time = time.time()
        self.time_taken = end_time - star_time
        return chunks

    def get_cost_and_time_taken(self):
        """
        Returns the time taken and cost for the last split operation.
        """
        return self.cost, self.time_taken