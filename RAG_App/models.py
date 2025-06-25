from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime


# --- Domain Models (using Pydantic for consistency and validation) ---

class DocumentChunk(BaseModel):
    """Domain model representing a chunk of a document"""
    id: str = ""
    content: str = ""
    document_id: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class Document(BaseModel):
    """Domain model representing a document in the RAG system"""
    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[DocumentChunk] = Field(default_factory=list)

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Add a chunk to this document"""
        self.chunks.append(chunk)
        chunk.document_id = self.id

class QueryResult(BaseModel):
    """Domain model representing the result of a RAG query"""
    query: str = ""
    answer: str = ""
    retrieved_documents: List[DocumentChunk] = Field(default_factory=list)
    rerank_explanation: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Flashcard(BaseModel):
    """Domain model representing a flashcard generated from documents"""
    id: str = ""
    question: str = ""
    answer: str = ""
    document_id: str = ""
    document_chunk_id: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvaluationResult(BaseModel):
    """Domain model representing evaluation metrics for a RAG query"""
    metrics: Dict[str, float] = Field(default_factory=dict)
    query: str = ""
    answer: str = ""
    ground_truth: str = ""
    retrieved_documents: List[DocumentChunk] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

# --- Configuration Models ---

class ComponentConfig(BaseModel):
    """Base model for a component's configuration."""
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class AppConfig(BaseModel):
    """Root model for the entire application configuration."""
    chunker: ComponentConfig
    embedder: ComponentConfig
    vector_store: ComponentConfig
    retriever: ComponentConfig
    reranker: ComponentConfig
    llm: ComponentConfig
    evaluator: ComponentConfig