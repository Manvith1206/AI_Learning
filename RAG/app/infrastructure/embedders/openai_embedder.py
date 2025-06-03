from typing import Dict, List, Any
import os
import openai
from app.core.interfaces.interfaces import Embedder


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI embedder
        
        Args:
            config: Configuration dictionary with model, etc.
        """
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = config.get("model", "text-embedding-ada-002")
        self.dimensions = config.get("dimensions", None)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of text documents to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        kwargs = {"model": self.model}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        
        response = self.client.embeddings.create(
            input=texts,
            **kwargs
        )
        
        return [data.embedding for data in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text
        
        Args:
            text: Query text to embed
        
        Returns:
            Embedding vector
        """
        kwargs = {"model": self.model}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        
        response = self.client.embeddings.create(
            input=[text],
            **kwargs
        )
        
        return response.data[0].embedding
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the embedder configuration
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
        self.model = config.get("model", self.model)
        self.dimensions = config.get("dimensions", self.dimensions)
