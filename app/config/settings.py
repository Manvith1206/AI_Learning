from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Define environment variables to be loaded.
    # Use sensible defaults where possible, or None if the var is mandatory.

    # API Keys
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    VOYAGE_API_KEY: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    JINA_RERANKER_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None # Anthropic API Key

    # Model Names (examples, to be expanded based on RAG_Constants.py)
    DEFAULT_CHAT_MODEL_NAME: str = "gemini-1.5-flash" # Example default
    DEFAULT_EMBEDDING_MODEL_NAME: str = "text-embedding-004" # Example default

    # Vector Store Configs
    CHROMA_COLLECTION_NAME: str = "chroma-rag-v1"
    PINECONE_INDEX_NAME: str = "test-rag-v1"
    # Add other store specific settings like host, port if needed
    # e.g. CHROMA_HOST: str = "localhost"
    # e.g. CHROMA_PORT: int = 8000

    # RAG Pipeline Defaults (examples)
    DEFAULT_TOP_K_RETRIEVAL: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7

    # Temporary file directory (if needed for uploads, etc.)
    TEMP_DOCS_DIR: str = "temp_docs"

    # Application Specific
    AI_APPLICATION_TITLE: str = "RAG System"


    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",          # Load from .env file if present
        env_file_encoding="utf-8",
        extra="ignore"            # Ignore extra fields from .env
    )

# Instantiate settings once
settings = Settings()
