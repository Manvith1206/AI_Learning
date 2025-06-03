from typing import Dict, List, Any
import uuid
from app.core.interfaces.interfaces import Chunker
from app.domain.models.models import Document, DocumentChunk

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("LangChain is not installed. Install it with 'pip install langchain'")


class RecursiveChunker(Chunker):
    """Recursive text chunker implementation using LangChain's RecursiveCharacterTextSplitter"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the recursive chunker
        
        Args:
            config: Configuration dictionary with chunk_size, chunk_overlap, etc.
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.separators = config.get("separators", ["\n\n", "\n", " ", ""])
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def split(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks
        
        Args:
            document: The document to split
        
        Returns:
            List of document chunks
        """
        # Split text into chunks
        texts = self.text_splitter.split_text(document.content)
        
        # Create DocumentChunk objects
        chunks = []
        for i, text in enumerate(texts):
            chunks.append(DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document.id,
                content=text,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(texts)
                }
            ))
        
        return chunks
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the chunker configuration
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", self.chunk_size)
        self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
        self.separators = config.get("separators", self.separators)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
