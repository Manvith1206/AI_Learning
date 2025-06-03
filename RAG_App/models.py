from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Document:
    """Domain model representing a document in the RAG system"""
    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List["DocumentChunk"] = field(default_factory=list)
    
    def add_chunk(self, chunk: "DocumentChunk") -> None:
        """Add a chunk to this document"""
        self.chunks.append(chunk)
        chunk.document_id = self.id


@dataclass
class DocumentChunk:
    """Domain model representing a chunk of a document"""
    id: str = ""
    content: str = ""
    document_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

@dataclass
class QueryResult:
    """Domain model representing the result of a RAG query"""
    query: str = ""
    answer: str = ""
    retrieved_documents: List[DocumentChunk] = field(default_factory=list)
    rerank_explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Flashcard:
    """Domain model representing a flashcard generated from documents"""
    id: str = ""
    question: str = ""
    answer: str = ""
    document_id: str = ""
    document_chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Domain model representing evaluation metrics for a RAG query"""
    metrics: Dict[str, float] = field(default_factory=dict)
    query: str = ""
    answer: str = ""
    ground_truth: str = ""
    retrieved_documents: List[DocumentChunk] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
