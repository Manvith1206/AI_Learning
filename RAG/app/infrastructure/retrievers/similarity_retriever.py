from typing import Dict, List, Any, Optional
from app.core.interfaces.interfaces import Retriever, VectorStore, Embedder
from app.domain.models.models import DocumentChunk


class SimilarityRetriever(Retriever):
    """Similarity-based retriever implementation"""
    
    def __init__(self, vector_store: VectorStore, embedder: Embedder, config: Dict[str, Any]):
        """
        Initialize the similarity retriever
        
        Args:
            vector_store: Vector store to retrieve from
            embedder: Embedder to create query embeddings
            config: Configuration dictionary with top_k, etc.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config
        self.top_k = config.get("top_k", 5)
        self.similarity_threshold = config.get("similarity_threshold", 0.0)
    
    def retrieve(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: Query string
            filter_dict: Optional filter dictionary
        
        Returns:
            List of retrieved document chunks
        """
        # Create query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Retrieve from vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filter_dict=filter_dict
        )
        
        # Filter by similarity threshold if specified
        if self.similarity_threshold > 0:
            results = [r for r in results if r["score"] >= self.similarity_threshold]
        
        # Convert to DocumentChunk objects
        chunks = []
        for result in results:
            chunks.append(DocumentChunk(
                id=result["id"],
                document_id=result["metadata"].get("document_id", ""),
                content=result["content"],
                chunk_index=result["metadata"].get("chunk_index", 0),
                metadata=result["metadata"],
                score=result["score"]
            ))
        
        return chunks
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the retriever configuration
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
        self.top_k = config.get("top_k", self.top_k)
        self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
