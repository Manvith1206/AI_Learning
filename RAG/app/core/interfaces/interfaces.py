from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple


class LLMService(ABC):
    """Interface for LLM services like OpenAI, Anthropic, Gemini, etc."""
    
    @abstractmethod
    def query(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Query the LLM with a prompt and return the response"""
        pass
    
    @abstractmethod
    def get_cost_and_time(self) -> Tuple[float, str]:
        """Return the cost and time metrics for this service"""
        pass


class VectorStore(ABC):
    """Interface for vector stores like Chroma, Pinecone, FAISS, etc."""
    
    @abstractmethod
    def store_embeddings(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Store document embeddings in the vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Search for similar documents based on query embedding"""
        pass
    
    @abstractmethod
    def get_cost_and_time(self) -> Tuple[float, str]:
        """Return the cost and time metrics for this service"""
        pass


class Embedder(ABC):
    """Interface for embedding models like OpenAI, Cohere, etc."""
    
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a query string"""
        pass
    
    @abstractmethod
    def get_cost_and_time(self) -> Tuple[float, str]:
        """Return the cost and time metrics for this service"""
        pass


class Chunker(ABC):
    """Interface for text chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks according to the chunking strategy"""
        pass
    
    @abstractmethod
    def get_cost_and_time(self) -> Tuple[float, str]:
        """Return the cost and time metrics for this service"""
        pass


class Retriever(ABC):
    """Interface for retrieval strategies"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        pass
    
    @abstractmethod
    def get_cost_and_time(self) -> Tuple[float, str]:
        """Return the cost and time metrics for this service"""
        pass


class Reranker(ABC):
    """Interface for reranking strategies"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
        """Rerank documents based on relevance to query and return explanation"""
        pass
    
    @abstractmethod
    def get_cost_and_time(self) -> Tuple[float, str]:
        """Return the cost and time metrics for this service"""
        pass


class Evaluator(ABC):
    """Interface for RAG evaluation strategies"""
    
    @abstractmethod
    def evaluate(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                 answer: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate RAG pipeline performance"""
        pass
    
    @abstractmethod
    def get_cost_and_time(self) -> Tuple[float, str]:
        """Return the cost and time metrics for this service"""
        pass


class DocumentRepository(ABC):
    """Interface for document storage and retrieval"""
    
    @abstractmethod
    def store_document(self, document: Dict[str, Any]) -> str:
        """Store a document and return its ID"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID"""
        pass
    
    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents"""
        pass


class ChatRepository(ABC):
    """Interface for chat history storage and retrieval"""
    
    @abstractmethod
    def store_message(self, message: Dict[str, Any]) -> str:
        """Store a chat message and return its ID"""
        pass
    
    @abstractmethod
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve chat history for a session"""
        pass
