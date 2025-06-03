from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

# You might want to define a common Document class/dataclass here or in models.py
# For now, we'll use Dict[str, Any] for document metadata and content.

class BaseVectorStore(ABC):
    """
    Abstract base class for Vector Store interactions.
    Defines a common interface for various vector database providers.
    """

    @abstractmethod
    def __init__(self, connection_string: Optional[str] = None, embedding_function: Optional[Any] = None, **kwargs: Any):
        """
        Initialize the Vector Store client.
        Args:
            connection_string (Optional[str]): Connection string or path for the vector store.
            embedding_function (Optional[Any]): The function or model used to generate embeddings.
                                                This might be an instance of BaseLLM or a specific embedding model.
            **kwargs: Additional provider-specific arguments.
        """
        pass

    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]], # Each dict could represent a document with 'text' and 'metadata'
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Add documents to the vector store.
        If embeddings are not provided, the store might try to generate them using its embedding_function.
        Args:
            documents (List[Dict[str, Any]]): A list of documents to add.
                                              Each document is a dictionary, potentially with 'page_content' and 'metadata'.
            embeddings (Optional[List[List[float]]]): Optional pre-computed embeddings for the documents.
            **kwargs: Additional provider-specific arguments.
        Returns:
            List[str]: A list of IDs for the added documents.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]: # List of documents with scores
        """
        Search for similar documents in the vector store.
        Args:
            query_embedding (List[float]): The embedding of the query.
            top_k (int): The number of similar documents to return.
            filter_criteria (Optional[Dict[str, Any]]): Filters to apply to the search (e.g., metadata filters).
            **kwargs: Additional provider-specific arguments.
        Returns:
            List[Dict[str, Any]]: A list of retrieved documents, possibly including their scores and metadata.
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str], **kwargs: Any) -> bool:
        """
        Delete documents from the vector store by their IDs.
        Args:
            document_ids (List[str]): A list of document IDs to delete.
            **kwargs: Additional provider-specific arguments.
        Returns:
            bool: True if deletion was successful (or partially successful), False otherwise.
        """
        pass

    @abstractmethod
    def update_document(self, document_id: str, document: Dict[str, Any], **kwargs: Any) -> bool:
        """
        Updates an existing document in the vector store.
        Args:
            document_id (str): The ID of the document to update.
            document (Dict[str, Any]): The new document content and metadata.
            **kwargs: Additional provider-specific arguments.
        Returns:
            bool: True if update was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_document_by_id(self, document_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieves a document by its ID.
        Args:
            document_id (str): The ID of the document to retrieve.
            **kwargs: Additional provider-specific arguments.
        Returns:
            Optional[Dict[str, Any]]: The document if found, else None.
        """
        pass

    # Consider other methods like:
    # - create_collection_if_not_exists
    # - list_collections
    # - get_collection_info
