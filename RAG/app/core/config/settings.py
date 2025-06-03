import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings loaded from environment variables with defaults"""
    
    # LLM Service settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_LLM_SERVICE: str = os.getenv("DEFAULT_LLM_SERVICE", "gemini")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-pro")
    
    # Vector store settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "rag-index")
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    DEFAULT_VECTOR_STORE: str = os.getenv("DEFAULT_VECTOR_STORE", "scikit_learn")
    
    # Embedder settings
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
    DEFAULT_EMBEDDER: str = os.getenv("DEFAULT_EMBEDDER", "gemini")
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "models/embedding-001")
    
    # Chunker settings
    DEFAULT_CHUNKER: str = os.getenv("DEFAULT_CHUNKER", "recursive")
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "150"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "70"))
    
    # Retriever settings
    DEFAULT_RETRIEVER: str = os.getenv("DEFAULT_RETRIEVER", "similarity")
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.0"))
    DEFAULT_KEYWORD_WEIGHT: float = float(os.getenv("DEFAULT_KEYWORD_WEIGHT", "0.3"))
    
    # Reranker settings
    DEFAULT_RERANKER: str = os.getenv("DEFAULT_RERANKER", "llm")
    DEFAULT_TOP_K_RERANK: int = int(os.getenv("DEFAULT_TOP_K_RERANK", "5"))
    
    # Evaluator settings
    DEFAULT_EVALUATOR: str = os.getenv("DEFAULT_EVALUATOR", "llm")
    
    # Application settings
    APP_NAME: str = os.getenv("APP_NAME", "RAG Modular")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"


settings = Settings()


def get_default_config() -> Dict[str, Dict[str, Any]]:
    """Get the default configuration for all components"""
    return {
        "chunker": {
            "type": settings.DEFAULT_CHUNKER,
            "params": {
                "chunk_size": settings.DEFAULT_CHUNK_SIZE,
                "chunk_overlap": settings.DEFAULT_CHUNK_OVERLAP
            }
        },
        "embedder": {
            "type": settings.DEFAULT_EMBEDDER,
            "params": {
                "model": settings.DEFAULT_EMBEDDING_MODEL
            }
        },
        "vector_store": {
            "type": settings.DEFAULT_VECTOR_STORE,
            "params": {
                "metric": "cosine"
            }
        },
        "retriever": {
            "type": settings.DEFAULT_RETRIEVER,
            "params": {
                "top_k": settings.DEFAULT_TOP_K,
                "similarity_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD
            }
        },
        "reranker": {
            "type": settings.DEFAULT_RERANKER,
            "params": {
                "top_k": settings.DEFAULT_TOP_K_RERANK
            }
        },
        "llm": {
            "type": settings.DEFAULT_LLM_SERVICE,
            "params": {
                "model": settings.DEFAULT_LLM_MODEL
            }
        },
        "evaluator": {
            "type": settings.DEFAULT_EVALUATOR,
            "params": {}
        }
    }
