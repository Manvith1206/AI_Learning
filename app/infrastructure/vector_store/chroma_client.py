from typing import List, Dict, Any, Optional
import uuid

from app.infrastructure.vector_store.base_store import BaseVectorStore

try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings, IDs, Metadatas, Where, GetResult
    CHROMA_AVAILABLE = True
except ImportError:
    print("chromadb library not found. Please install it.")
    CHROMA_AVAILABLE = False
    EmbeddingFunction = Any # type: ignore
    chromadb = None # type: ignore

class ChromaClient(BaseVectorStore):
    def __init__(self,
                 path: Optional[str] = "./chroma_data",
                 collection_name: str = "default_rag_collection",
                 # embedding_function: Optional[EmbeddingFunction] = None, # Chroma can take its own embed func
                 **kwargs: Any):
        super().__init__() # Call parent __init__
        if not CHROMA_AVAILABLE or chromadb is None:
            raise ImportError("chromadb library is required to use ChromaClient.")

        # For embedding_function, if you want Chroma to handle embeddings,
        # you'd typically pass a chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
        # or similar. However, our BaseVectorStore design expects embeddings to be passed to add_documents.
        # So, we won't use Chroma's internal embedding function here to align with BaseVectorStore.

        if path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            # Ephemeral client, useful for testing or in-memory operations
            self.client = chromadb.Client()

        try:
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e: # Catch potential db connection or creation errors
            print(f"Error initializing Chroma collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to initialize Chroma collection: {e}") from e

        self.collection_name = collection_name
        print(f"ChromaClient initialized. Collection '{collection_name}' at path '{path if path else 'in-memory'}'. Documents: {self.collection.count()}")


    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any
    ) -> List[str]:
        if embeddings is None:
            # This aligns with our BaseVectorStore, where DocumentService generates embeddings.
            raise ValueError("ChromaClient, as used in this system, requires pre-computed embeddings for add_documents.")
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must be the same.")

        ids: List[str] = []
        docs_to_store: Documents = [] # List[str]
        metadatas_to_store: List[Metadatas] = [] # List[Dict[str, Any]]

        for doc_idx, doc_data in enumerate(documents):
            doc_id = doc_data.get("id", str(uuid.uuid4()))
            ids.append(doc_id)

            page_content = doc_data.get("page_content")
            if not isinstance(page_content, str):
                raise ValueError(f"Document 'page_content' must be a string for document at index {doc_idx} (ID suggestion: {doc_id})")
            docs_to_store.append(page_content)

            metadata = doc_data.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(f"Document 'metadata' must be a dict for document at index {doc_idx} (ID suggestion: {doc_id})")
            metadatas_to_store.append(metadata or {}) # Ensure metadata is a dict, even if None was passed

        try:
            # Chroma's upsert is suitable here: adds if new ID, updates if ID exists.
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs_to_store, metadatas=metadatas_to_store)
            return ids
        except Exception as e:
            print(f"Error upserting documents to Chroma collection '{self.collection_name}': {e}")
            raise RuntimeError(f"Chroma DB upsert failed: {e}") from e

    def search(
        self, query_embedding: List[float], top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None, # This is Chroma's 'where'
        include: Optional[List[str]] = None, # Chroma's 'include'
        **kwargs: Any
    ) -> List[Dict[str, Any]]:

        query_include_params = include or ["documents", "metadatas", "distances"]

        try:
            results: GetResult = self.collection.query(
                query_embeddings=[query_embedding], # Chroma expects a list of embeddings
                n_results=min(top_k, self.collection.count()), # Cannot request more than in collection
                where=filter_criteria, # type: ignore # Chroma's Where type is specific, but Dict is often used
                include=query_include_params
            )
        except Exception as e: # Catch potential query errors (e.g., invalid filter)
            print(f"Error searching Chroma collection '{self.collection_name}': {e}")
            raise RuntimeError(f"Chroma DB query failed: {e}") from e

        formatted_results: List[Dict[str, Any]] = []
        # Chroma returns lists of lists for each field, one inner list per query embedding (we send one)
        res_ids = results.get('ids', [[]])[0]
        res_docs_content = results.get('documents', [[]])[0] if results.get('documents') else [[]]
        res_metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else [[]]
        res_distances = results.get('distances', [[]])[0] if results.get('distances') else [[]]

        for i in range(len(res_ids)):
            doc_id = res_ids[i]
            # Ensure content, metadata, and distance are accessed safely
            content = res_docs_content[i] if res_docs_content and i < len(res_docs_content) else None
            metadata = res_metadatas[i] if res_metadatas and i < len(res_metadatas) else None
            distance = res_distances[i] if res_distances and i < len(res_distances) else None

            similarity_score = (1.0 / (1.0 + distance)) if distance is not None else 0.0 # Convert distance to similarity

            formatted_results.append({
                "id": doc_id,
                "page_content": content,
                "metadata": metadata or {}, # Ensure metadata is always a dict
                "score": similarity_score
            })
        return formatted_results

    def delete_documents(self, document_ids: List[str], **kwargs: Any) -> bool:
        if not document_ids: return True
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception as e:
            print(f"Error deleting documents from Chroma collection '{self.collection_name}': {e}")
            # Depending on requirements, you might want to return False or re-raise
            raise RuntimeError(f"Chroma DB delete failed: {e}") from e


    def update_document(self, document_id: str, document: Dict[str, Any], embedding: Optional[List[float]] = None, **kwargs: Any) -> bool:
        # Chroma's upsert handles updates naturally if the ID exists.
        # This method aligns with BaseVectorStore, but might be a bit redundant for Chroma.

        page_content = document.get("page_content")
        metadata = document.get("metadata")

        if page_content is None and metadata is None and embedding is None:
            print(f"Warning: Update for document ID '{document_id}' called with no content, metadata, or embedding to update.")
            return False # Or True, as nothing changed.

        # For Chroma, we need all fields for an upsert if we are to "update"
        # If embedding is not provided, but content is, it's problematic for our design
        # where DocumentService handles embedding generation.
        if embedding is None and page_content is not None:
             print(f"Warning: Content for ID '{document_id}' is being updated, but no new embedding was provided. This may lead to inconsistencies if content changed.")
             # Ideally, if content changes, a new embedding should be generated by DocumentService and passed here.
             # If only metadata is updated, existing embedding is fine.

        # To perform an update, we essentially upsert. We need the existing doc's parts if not provided.
        # However, BaseVectorStore's update_document doesn't require fetching the old doc.
        # We will proceed with upserting only the provided fields.

        upsert_embeddings = [embedding] if embedding else None
        upsert_documents = [page_content] if page_content is not None else None
        upsert_metadatas = [metadata if metadata is not None else {}] # Ensure metadata is a dict

        try:
            self.collection.upsert(
                ids=[document_id],
                embeddings=upsert_embeddings, # type: ignore # Chroma types are strict, allow None if content/meta updated
                documents=upsert_documents,   # type: ignore
                metadatas=upsert_metadatas    # type: ignore
            )
            return True
        except Exception as e:
            print(f"Error updating document ID '{document_id}' in Chroma: {e}")
            raise RuntimeError(f"Chroma DB update (via upsert) failed: {e}") from e


    def get_document_by_id(self, document_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        try:
            # include must be a list of strings from Literal["documents", "embeddings", "metadatas", "distances"]
            result: GetResult = self.collection.get(ids=[document_id], include=["documents", "metadatas"])
        except Exception as e: # More specific exceptions could be caught, e.g., if ID not found (though Chroma may not raise for that)
            print(f"Error fetching document ID '{document_id}' from Chroma: {e}")
            return None

        if not result or not result.get('ids') or not result['ids'][0]: # Check if ID was actually found
            return None

        # Assuming result['ids'], result['documents'], result['metadatas'] are lists of the same length (1 in this case)
        doc_content = result['documents'][0] if result.get('documents') and result['documents'] else None
        doc_metadata = result['metadatas'][0] if result.get('metadatas') and result['metadatas'] else {}

        return {
            "id": result['ids'][0],
            "page_content": doc_content,
            "metadata": doc_metadata
        }

    def clear_collection(self) -> bool:
        """Deletes all entries from the current collection and recreates it."""
        try:
            collection_name_to_clear = self.collection.name
            count = self.collection.count()
            self.client.delete_collection(name=collection_name_to_clear)
            print(f"Successfully deleted collection: {collection_name_to_clear} (had {count} items).")
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(name=collection_name_to_clear)
            print(f"Successfully recreated empty collection: {collection_name_to_clear}. New count: {self.collection.count()}")
            return True
        except Exception as e:
            print(f"Error clearing collection {self.collection_name}: {e}")
            # Attempt to recreate just in case deletion was partial or client state is odd
            try:
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                print(f"Ensured collection '{self.collection_name}' exists after clear error.")
            except Exception as e_rec:
                 print(f"Failed to even recreate collection '{self.collection_name}' after clear error: {e_rec}")
            return False
