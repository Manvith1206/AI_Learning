from typing import Dict, List, Any, Optional
import json
import os
import uuid
from datetime import datetime
from app.core.interfaces.interfaces import DocumentRepository
from app.domain.models.models import Document


class FileSystemDocumentRepository(DocumentRepository):
    """File system implementation of document repository"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file system document repository
        
        Args:
            config: Configuration dictionary with storage_path, etc.
        """
        self.config = config
        self.storage_path = config.get("storage_path", "./documents")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Index of documents
        self.index_path = os.path.join(self.storage_path, "index.json")
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load document index from file"""
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save document index to file"""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)
    
    def add(self, document: Document) -> str:
        """
        Add a document to the repository
        
        Args:
            document: Document to add
        
        Returns:
            ID of the added document
        """
        # Generate ID if not present
        doc_id = document.id if document.id else str(uuid.uuid4())
        document.id = doc_id
        
        # Save document content
        doc_path = os.path.join(self.storage_path, f"{doc_id}.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(document.content)
        
        # Update index
        self.index[doc_id] = {
            "id": doc_id,
            "title": document.metadata.get("title", "Untitled"),
            "filename": document.metadata.get("filename", "unknown.txt"),
            "created_at": document.metadata.get("created_at", datetime.now().isoformat()),
            "file_path": doc_path,
            "metadata": document.metadata
        }
        
        self._save_index()
        return doc_id
    
    def get(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document if found, None otherwise
        """
        if doc_id not in self.index:
            return None
        
        doc_info = self.index[doc_id]
        doc_path = doc_info["file_path"]
        
        if not os.path.exists(doc_path):
            return None
        
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return Document(
            id=doc_id,
            content=content,
            metadata=doc_info["metadata"]
        )
    
    def get_all(self) -> List[Document]:
        """
        Get all documents
        
        Returns:
            List of all documents
        """
        documents = []
        
        for doc_id in self.index:
            doc = self.get(doc_id)
            if doc:
                documents.append(doc)
        
        return documents
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            True if document was deleted, False otherwise
        """
        if doc_id not in self.index:
            return False
        
        doc_info = self.index[doc_id]
        doc_path = doc_info["file_path"]
        
        # Delete file if it exists
        if os.path.exists(doc_path):
            os.remove(doc_path)
        
        # Remove from index
        del self.index[doc_id]
        self._save_index()
        
        return True
    
    def update(self, document: Document) -> bool:
        """
        Update a document
        
        Args:
            document: Document to update
        
        Returns:
            True if document was updated, False otherwise
        """
        if not document.id or document.id not in self.index:
            return False
        
        # Update document content
        doc_path = os.path.join(self.storage_path, f"{document.id}.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(document.content)
        
        # Update index
        self.index[document.id].update({
            "title": document.metadata.get("title", self.index[document.id].get("title", "Untitled")),
            "metadata": document.metadata
        })
        
        self._save_index()
        return True
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the repository configuration
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
        new_storage_path = config.get("storage_path")
        
        if new_storage_path and new_storage_path != self.storage_path:
            self.storage_path = new_storage_path
            os.makedirs(self.storage_path, exist_ok=True)
            self.index_path = os.path.join(self.storage_path, "index.json")
            self.index = self._load_index()
