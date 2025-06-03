from abc import ABC, abstractmethod
import time, re
from typing import List, Dict, Any, Optional
try: from langchain.text_splitter import TokenTextSplitter
except ImportError: TokenTextSplitter = None
try: from sentence_transformers import SentenceTransformer
except ImportError: SentenceTransformer = None
try: import numpy as np
except ImportError: np = None
try:
    from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
except ImportError: LangchainSemanticChunker, FastEmbedEmbeddings = None, None

class BaseChunker(ABC):
    def __init__(self): self.time_taken, self.cost = 0.0, 0.0
    @abstractmethod
    def split_text(self, text: str) -> List[str]: pass
    def get_cost_and_time_taken(self) -> tuple[float, float]: return self.cost, self.time_taken

class PageChunker(BaseChunker):
    def split_text(self, text: str) -> List[str]:
        start_time = time.time()
        if not text: return []
        pages = re.split(r"--- Page \d+:.*?---\n", text)
        self.time_taken = time.time() - start_time
        return [p.strip() for p in pages if p.strip()]

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int=600, chunk_overlap: int=200, encoding_name: str="cl100k_base"):
        super().__init__()
        if TokenTextSplitter is None: raise ImportError("Langchain's TokenTextSplitter required.")
        self.text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name=encoding_name)
    def split_text(self, text: str) -> List[str]:
        start_time = time.time()
        chunks = self.text_splitter.split_text(text=text) if text else []
        self.time_taken, self.cost = time.time() - start_time, 0
        return chunks

class SentenceChunker(BaseChunker):
    def __init__(self, max_sentences: int = 5): super().__init__(); self.max_sentences = max_sentences
    def split_text(self, text: str) -> List[str]:
        start_time = time.time()
        if not text: return []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = [' '.join(sentences[i:i+self.max_sentences]) for i in range(0, len(sentences), self.max_sentences)]
        self.time_taken, self.cost = time.time() - start_time, 0
        return chunks

class SemanticChunker(BaseChunker):
    def __init__(self, model_name: str="all-MiniLM-L6-v2", similarity_threshold: float=0.7, min_chunk_size: int=3, max_chunk_size: int=20, max_sentences_for_segmentation: int=300):
        super().__init__()
        if SentenceTransformer is None: raise ImportError("sentence-transformers required.")
        if np is None: raise ImportError("numpy required.")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold, self.min_chunk_size, self.max_chunk_size, self.max_sentences_for_segmentation = similarity_threshold, min_chunk_size, max_chunk_size, max_sentences_for_segmentation
    def _cosine_similarity(self, v1: Any, v2: Any) -> float: return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    def _preprocess_text(self, text: str) -> str: return re.sub(r'\n{3,}', '\n\n', re.sub(r'\s+', ' ', text)).strip()
    def _segment_into_sentences(self, text: str) -> List[str]:
        s = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', self._preprocess_text(text))
        fs = []; [fs.extend(sp for sp in re.split(r'(?<=[,;])\s+', sen) if sp.strip()) if len(sen) > self.max_sentences_for_segmentation else (fs.append(sen) if sen.strip() else None) for sen in s]
        return fs
    def _find_chunk_boundaries(self, embeddings: Any) -> List[int]:
        b, ccs = [], 0
        for i in range(len(embeddings) - 1):
            ccs += 1
            sim = float(self._cosine_similarity(embeddings[i], embeddings[i+1]))
            if (sim < self.similarity_threshold and ccs >= self.min_chunk_size) or ccs >= self.max_chunk_size: b.append(i+1); ccs=0
        if not b or b[-1] != len(embeddings): b.append(len(embeddings))
        return b
    def split_text(self, text: str) -> List[str]:
        start_time = time.time()
        sentences = self._segment_into_sentences(text)
        if not sentences: return []
        embeddings = self.model.encode(sentences)
        boundaries = self._find_chunk_boundaries(embeddings)
        chunks, start_idx = [], 0
        for end_idx in boundaries: chunks.append(' '.join(sentences[start_idx:end_idx])); start_idx = end_idx
        self.time_taken, self.cost = time.time() - start_time, 0
        return chunks

class SemanticChunkerWithLangChain(BaseChunker):
    def __init__(self, embedding_model_name: str="BAAI/bge-base-en-v1.5", breakpoint_threshold_type="percentile"):
        super().__init__()
        if LangchainSemanticChunker is None: raise ImportError("langchain_experimental required.")
        # Ensure FastEmbedEmbeddings is available and used correctly
        if FastEmbedEmbeddings is None: raise ImportError("langchain_community.embeddings.fastembed required for SemanticChunkerWithLangChain")
        self.semantic_chunker = LangchainSemanticChunker(FastEmbedEmbeddings(model_name=embedding_model_name), breakpoint_threshold_type=breakpoint_threshold_type)
    def split_text(self, text: str) -> List[str]:
        start_time = time.time()
        chunks = self.semantic_chunker.split_text(text=text) if text else []
        self.time_taken, self.cost = time.time() - start_time, 0
        return chunks

def get_chunker(chunker_type: str="recursive", params: Optional[Dict[str,Any]]=None) -> BaseChunker:
    p = params or {}
    if chunker_type == "recursive": return RecursiveChunker(**p)
    if chunker_type == "sentence": return SentenceChunker(**p)
    if chunker_type == "semantic": return SemanticChunker(**p)
    if chunker_type == "page": return PageChunker()
    if chunker_type == "semantic_langchain": return SemanticChunkerWithLangChain(**p)
    raise ValueError(f"Unsupported chunker type: {chunker_type}")
