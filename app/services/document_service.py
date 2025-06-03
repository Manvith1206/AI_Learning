from app.infrastructure.vector_store.base_store import BaseVectorStore
from app.infrastructure.llm.base_llm import BaseLLM
from app.config import settings
from typing import List, Dict, Any, Optional, Union, IO
import os
import uuid
import logging

from .document_loaders import get_loader
from .chunkers import get_chunker
from .retrievers import BaseRetriever
from .rerankers import BaseReranker

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self,
                 vector_store: BaseVectorStore,
                 embedding_client: BaseLLM,
                 retriever: BaseRetriever,
                 reranker: Optional[BaseReranker] = None):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.retriever = retriever
        self.reranker = reranker
        self._all_document_texts_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_is_dirty = True

    def _load_raw_text_from_file(self, file_path: str, file_type: Optional[str] = None) -> str:
        logger.info(f"Loading text from file: {file_path}, type: {file_type}")
        try:
            loader = get_loader(file_path, file_type)
            return loader.load_document(file_path)
        except ValueError as e:
            logger.error(f"Error getting loader or loading document {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading document {file_path}: {e}")
            raise RuntimeError(f"Could not load document {file_path}") from e

    def _chunk_text(self, text: str, chunker_type: str = "recursive", chunker_params: Optional[Dict[str, Any]] = None) -> List[str]:
        logger.info(f"Chunking text using type: {chunker_type} with params: {chunker_params}")
        try:
            chunker_params = chunker_params or {}
            chunker = get_chunker(chunker_type, params=chunker_params)
            return chunker.split_text(text)
        except ValueError as e:
            logger.error(f"Error getting or using chunker {chunker_type}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during chunking with {chunker_type}: {e}")
            raise RuntimeError(f"Could not chunk text using {chunker_type}") from e

    def process_uploaded_file(
        self,
        uploaded_file: IO[bytes],
        file_name: str,
        chunker_type: str = "recursive",
        chunker_params: Optional[Dict[str, Any]] = None,
        metadata_override: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        temp_dir = settings.TEMP_DOCS_DIR
        os.makedirs(temp_dir, exist_ok=True)
        unique_id = str(uuid.uuid4())
        # Use a safe version of the original filename for the temp file, prefixed with unique_id
        safe_original_filename = os.path.basename(file_name)
        temp_file_name = f"{unique_id}_{safe_original_filename}"
        temp_file_path = os.path.join(temp_dir, temp_file_name)
        try:
            with open(temp_file_path, "wb") as f: f.write(uploaded_file.read())
            logger.info(f"Temporarily saved uploaded file to {temp_file_path}")
            raw_text = self._load_raw_text_from_file(temp_file_path)
            if not raw_text or raw_text.isspace(): # Check for empty or whitespace-only text
                logger.warning(f"No text extracted from {file_name} or text is empty/whitespace."); return []
            chunks = self._chunk_text(raw_text, chunker_type, chunker_params)
            if not chunks: # Check if chunking produced any output
                logger.warning(f"Chunking of {file_name} resulted in no chunks.")
                return []
            logger.info(f"Successfully chunked {file_name} into {len(chunks)} chunks.")
            processed_documents: List[Dict[str, Any]] = []
            base_metadata = {"source_filename": file_name, **(metadata_override or {})}
            for i, chunk_text in enumerate(chunks):
                doc_id = str(uuid.uuid4())
                processed_documents.append({
                    "id": doc_id, "page_content": chunk_text,
                    "metadata": {**base_metadata, "chunk_index": i, "original_doc_id": unique_id}
                })
            return processed_documents
        finally:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path); logger.info(f"Removed temp file: {temp_file_path}")
                except OSError as e: logger.error(f"Error removing temp file {temp_file_path}: {e}")

    def add_documents_to_store(self, documents: List[Dict[str, Any]]) -> List[str]:
        if not documents:
            logger.info("No documents provided to add_documents_to_store.")
            return []
        texts_to_embed = [doc["page_content"] for doc in documents]
        if not self.embedding_client:
            logger.error("Embedding client not configured in DocumentService.")
            raise ValueError("Embedding client not configured.")
        logger.info(f"Generating embeddings for {len(texts_to_embed)} document chunks.")
        try: embeddings = self.embedding_client.generate_embeddings(texts=texts_to_embed)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise RuntimeError("Embedding generation failed.") from e
        if len(embeddings) != len(documents):
            logger.error(f"Mismatch between number of documents ({len(documents)}) and generated embeddings ({len(embeddings)}).")
            raise RuntimeError("Embedding count mismatch.")
        logger.info(f"Adding {len(documents)} documents to vector store.")
        try:
            document_ids = self.vector_store.add_documents(documents=documents, embeddings=embeddings)
            logger.info(f"Successfully added documents. IDs: {document_ids}")

            # Update BM25 cache
            self._cache_is_dirty = True # Mark dirty, will be clean after successful update
            if self._all_document_texts_cache is None: self._all_document_texts_cache = []

            # Efficiently update or add to cache
            temp_cache_map = {d["id"]: d for d in self._all_document_texts_cache}
            for doc in documents:
                temp_cache_map[doc["id"]] = {"id": doc["id"], "page_content": doc["page_content"]}
            self._all_document_texts_cache = list(temp_cache_map.values())
            self._cache_is_dirty = False # Cache is now up-to-date

            return document_ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
            self._cache_is_dirty = True # Cache might be inconsistent if add failed
            raise RuntimeError("Failed to add documents to vector store.") from e

    def _get_all_document_texts_for_bm25(self) -> Optional[List[Dict[str, Any]]]:
        # This method assumes the cache is managed by add/ingest/clear operations.
        # It's primarily for HybridRetriever.
        if self._cache_is_dirty:
            logger.warning("BM25 cache is marked as dirty. Consider rebuilding or ensuring it's up-to-date if issues occur.")
        if self._all_document_texts_cache is None:
            logger.warning("BM25 document cache is None. Hybrid retriever might not function optimally without it.")
        return self._all_document_texts_cache

    def retrieve_relevant_documents(
        self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self.embedding_client:
            logger.error("Embedding client not configured in DocumentService for retrieval.")
            raise ValueError("Embedding client required for document retrieval.")
        logger.info(f"Generating embedding for query: '{query[:50]}...'")
        try: query_embedding = self.embedding_client.generate_embeddings(texts=[query])[0]
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}", exc_info=True)
            raise RuntimeError("Query embedding failed.") from e

        effective_top_k = top_k if top_k is not None else settings.DEFAULT_TOP_K_RETRIEVAL
        retriever_kwargs = {"filter_criteria": filters}

        # Special handling for HybridRetriever's need for all_document_texts
        if self.retriever.__class__.__name__ == "HybridRetriever":
            all_texts_for_bm25 = self._get_all_document_texts_for_bm25()
            if all_texts_for_bm25 is None:
                logger.warning("HybridRetriever is used, but the BM25 document cache is not populated. BM25 may not work correctly.")
            retriever_kwargs["all_document_texts"] = all_texts_for_bm25

        logger.info(f"Retrieving documents via: {self.retriever.__class__.__name__} (top_k={effective_top_k}, filters={filters})")
        try:
            retrieved_docs = self.retriever.retrieve(
                query_embedding=query_embedding, query_text=query,
                vector_store=self.vector_store, top_k=effective_top_k, **retriever_kwargs
            )
            logger.info(f"Retrieved {len(retrieved_docs)} documents initially.")
        except Exception as e:
            logger.error(f"Initial document retrieval failed: {e}", exc_info=True)
            raise RuntimeError("Document retrieval failed.") from e

        if self.reranker and retrieved_docs:
            logger.info(f"Reranking {len(retrieved_docs)} documents via: {self.reranker.__class__.__name__}")
            try:
                reranker_kwargs = {} # Add any specific reranker args if needed from settings or params
                reranked_docs, explanation = self.reranker.rerank(
                    query=query, documents=retrieved_docs, top_k=effective_top_k, **reranker_kwargs
                )
                if explanation: logger.info(f"Reranking explanation: {explanation}")
                logger.info(f"Reranked into {len(reranked_docs)} documents.")
                return reranked_docs
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Returning initially retrieved documents.", exc_info=True)
                return retrieved_docs # Fallback to pre-reranked docs
        return retrieved_docs

    def ingest_file(
        self, uploaded_file: IO[bytes], file_name: str, chunker_type: str="recursive",
        chunker_params: Optional[Dict[str,Any]]=None, metadata_override: Optional[Dict[str,Any]]=None
    ) -> List[str]:
        logger.info(f"Starting ingestion process for file: {file_name} with chunker: {chunker_type}")
        processed_docs = self.process_uploaded_file(
            uploaded_file, file_name, chunker_type, chunker_params, metadata_override
        )
        if not processed_docs:
            logger.warning(f"No documents were processed from file {file_name}, nothing to add to store.")
            return []

        # This call will add to store and also update the BM25 cache internally
        added_doc_ids = self.add_documents_to_store(documents=processed_docs)
        logger.info(f"Successfully ingested file {file_name} and added {len(added_doc_ids)} document chunks to store.")
        return added_doc_ids

    def clear_documents_and_cache(self):
        logger.info("Clearing all document texts cache for BM25.")
        self._all_document_texts_cache = None
        self._cache_is_dirty = True # Mark as dirty since it's now empty

        logger.info("Attempting to clear documents from the vector store.")
        try:
            # This is a generic attempt. Specific vector stores might need different methods.
            # For example, some might require deleting by IDs (fetch all IDs then delete)
            # or have a dedicated "delete_all" or "clear_collection" method.
            if hasattr(self.vector_store, 'delete_documents') and callable(getattr(self.vector_store, 'delete_documents')):
                # If we had a way to get all document IDs from the store, we could use it here.
                # For now, this part is more of a placeholder for a robust clear.
                # Example: if self._all_document_texts_cache (before clearing) held all doc IDs:
                # all_ids = [doc['id'] for doc in previously_cached_docs]
                # if all_ids: self.vector_store.delete_documents(document_ids=all_ids)
                logger.warning("Generic vector_store.delete_documents exists, but requires IDs. Full clear might need specific implementation or all IDs.")

            elif hasattr(self.vector_store, 'clear_collection') and callable(getattr(self.vector_store, 'clear_collection')):
                 self.vector_store.clear_collection() # type: ignore
                 logger.info("Successfully called clear_collection on vector store.")
            elif hasattr(self.vector_store, '_collection') and hasattr(getattr(self.vector_store, '_collection'), 'delete'): # Chroma specific
                # This is a more direct way for Chroma, but relies on internal structure
                # collection = getattr(self.vector_store, '_collection')
                # all_ids_in_collection = collection.get(include=[])['ids'] # Get all IDs
                # if all_ids_in_collection:
                #    collection.delete(ids=all_ids_in_collection)
                #    logger.info(f"Cleared Chroma collection by deleting {len(all_ids_in_collection)} entries.")
                # else:
                #    logger.info("Chroma collection was already empty or get IDs failed.")
                logger.warning("Detected Chroma-like structure but direct clear via _collection.delete is risky. Prefer a high-level clear_collection if available.")
            else:
                logger.warning("Vector store does not have a clear_collection or recognized delete_documents method for a full clear.")
        except Exception as e:
            logger.error(f"Error trying to clear vector store: {e}", exc_info=True)
