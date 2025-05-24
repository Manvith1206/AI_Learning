from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from rag_modular.Chunkers.base_chunker import BaseChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

class SemanticChunkerWithLangChain(BaseChunker):
    def __init__(self):
        self.embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.semantic_chunker = SemanticChunker(self.embed_model, breakpoint_threshold_type="percentile")

    def split_text(self, text):
        chunks = self.semantic_chunker.split_text(text=text)
        
        return chunks
