from typing import Dict, List, Any, Optional
import uuid
import numpy as np
from app.core.interfaces.interfaces import VectorStore
from app.domain.models.models import DocumentChunk

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("ChromaDB is not installed. Install it with 'pip install chromadb'")


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ChromaDB vector store
        
        Args:
            config: Configuration dictionary with persistence path, etc.
        """
        self.config = config
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        self.collection_name = config.get("collection_name", "documents")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except ValueError:
            self.collection = self.client.create_collection(name=self.collection_name)
    
    def add(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> List[str]:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks to add
            embeddings: List of embeddings for each chunk
        
        Returns:
            List of IDs for the added chunks
        """
        if not chunks:
            return []
        
        # Generate IDs if not present
        ids = [chunk.id if chunk.id else str(uuid.uuid4()) for chunk in chunks]
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        # Extract metadata
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            } 
            for chunk in chunks
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return ids
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filter dictionary
        
        Returns:
            List of search results with id, content, metadata, and score
        """
        # Execute query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": float(results["distances"][0][i]) if "distances" in results else None
                })
        
        return formatted_results
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete chunks from the vector store
        
        Args:
            ids: List of chunk IDs to delete
        """
        self.collection.delete(ids=ids)
    
    def clear(self) -> None:
        """Clear all chunks from the vector store"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the vector store configuration
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
        
        # If collection name changes, need to get/create the new collection
        new_collection_name = config.get("collection_name")
        if new_collection_name and new_collection_name != self.collection_name:
            self.collection_name = new_collection_name
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except ValueError:
                self.collection = self.client.create_collection(name=self.collection_name)
