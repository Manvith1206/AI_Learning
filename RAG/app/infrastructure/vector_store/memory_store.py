from typing import Dict, List, Any, Optional
import uuid
import numpy as np
from app.core.interfaces import VectorStore
from app.domain.models import DocumentChunk


class InMemoryVectorStore(VectorStore):
    """In-memory vector store implementation for testing and development"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the in-memory vector store
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunks = {}  # id -> chunk
        self.embeddings = {}  # id -> embedding
        self.metadata = {}  # id -> metadata
    
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
        
        # Store chunks and embeddings
        for i, chunk_id in enumerate(ids):
            self.chunks[chunk_id] = chunks[i].content
            self.embeddings[chunk_id] = embeddings[i]
            self.metadata[chunk_id] = {
                "document_id": chunks[i].document_id,
                "chunk_index": chunks[i].chunk_index,
                **chunks[i].metadata
            }
        
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
        if not self.embeddings:
            return []
        
        # Convert query embedding to numpy array
        query_embedding_np = np.array(query_embedding)
        
        # Calculate cosine similarity for all embeddings
        similarities = {}
        for chunk_id, embedding in self.embeddings.items():
            # Apply filter if provided
            if filter_dict and not self._matches_filter(self.metadata[chunk_id], filter_dict):
                continue
                
            # Calculate cosine similarity
            embedding_np = np.array(embedding)
            similarity = np.dot(query_embedding_np, embedding_np) / (
                np.linalg.norm(query_embedding_np) * np.linalg.norm(embedding_np)
            )
            similarities[chunk_id] = similarity
        
        # Sort by similarity and get top-k
        sorted_ids = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)[:top_k]
        
        # Format results
        results = []
        for chunk_id in sorted_ids:
            results.append({
                "id": chunk_id,
                "content": self.chunks[chunk_id],
                "metadata": self.metadata[chunk_id],
                "score": float(similarities[chunk_id])
            })
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter dictionary"""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete chunks from the vector store
        
        Args:
            ids: List of chunk IDs to delete
        """
        for chunk_id in ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                del self.embeddings[chunk_id]
                del self.metadata[chunk_id]
    
    def clear(self) -> None:
        """Clear all chunks from the vector store"""
        self.chunks = {}
        self.embeddings = {}
        self.metadata = {}
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the vector store configuration
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
