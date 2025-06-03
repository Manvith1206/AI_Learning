from app.infrastructure.vector_store.base_store import BaseVectorStore
from app.infrastructure.llm.base_llm import BaseLLM
from app.config import settings
from typing import List, Dict, Any, Optional, IO
import os, uuid, logging
from .document_loaders import get_loader
from .chunkers import get_chunker

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self, vector_store: BaseVectorStore, embedding_client: Optional[BaseLLM] = None):
        self.vector_store, self.embedding_client = vector_store, embedding_client

    def _load_raw_text_from_file(self, file_path: str, file_type: Optional[str]=None) -> str:
        logger.info(f"Loading text from file: {file_path}, type: {file_type}")
        try: return get_loader(file_path, file_type).load_document(file_path)
        except ValueError as e: logger.error(f"Error getting loader/loading {file_path}: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error loading {file_path}: {e}"); raise RuntimeError(f"Could not load {file_path}") from e

    def _chunk_text(self, text: str, chunker_type: str="recursive", chunker_params: Optional[Dict[str,Any]]=None) -> List[str]:
        logger.info(f"Chunking text using type: {chunker_type} with params: {chunker_params}")
        try: return get_chunker(chunker_type, params=chunker_params or {}).split_text(text)
        except ValueError as e: logger.error(f"Error get/use chunker {chunker_type}: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error chunking {chunker_type}: {e}"); raise RuntimeError(f"Could not chunk with {chunker_type}") from e

    def process_uploaded_file(self, uploaded_file: IO[bytes], file_name: str, chunker_type: str="recursive", chunker_params: Optional[Dict[str,Any]]=None, metadata_override: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
        temp_dir = settings.TEMP_DOCS_DIR; os.makedirs(temp_dir, exist_ok=True)
        # Ensure filename is safe and create a unique name if needed, though original name is good for metadata
        safe_file_name = os.path.basename(file_name) # Basic sanitization
        temp_file_path = os.path.join(temp_dir, safe_file_name)
        try:
            with open(temp_file_path, "wb") as f: f.write(uploaded_file.read())
            logger.info(f"Temporarily saved to {temp_file_path}")
            raw_text = self._load_raw_text_from_file(temp_file_path)
            if not raw_text or raw_text.isspace():
                logger.warning(f"No text extracted from {safe_file_name} or text is empty/whitespace.")
                return []
            chunks = self._chunk_text(raw_text, chunker_type, chunker_params)
            logger.info(f"Chunked {safe_file_name} into {len(chunks)} chunks.")
            if not chunks:
                logger.warning(f"Chunking resulted in no chunks for {safe_file_name}.")
                return []
            processed_docs = []
            base_meta = {"source": safe_file_name, **(metadata_override or {})}
            for i, chunk_text in enumerate(chunks):
                processed_docs.append({"id":str(uuid.uuid4()), "page_content":chunk_text, "metadata":{**base_meta, "chunk_index":i}})
            return processed_docs
        except Exception as e:
            logger.error(f"Error processing uploaded file {safe_file_name}: {e}", exc_info=True)
            # Depending on desired error handling, could re-raise, or return empty / specific error structure
            raise # Re-raise for now to make issues visible
        finally:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path); logger.info(f"Removed temp file: {temp_file_path}")
                except OSError as e: logger.error(f"Error removing temp {temp_file_path}: {e}")

    def add_documents_to_store(self, documents: List[Dict[str,Any]]) -> List[str]:
        if not documents:
            logger.info("No documents provided to add_documents_to_store.")
            return []
        texts = [doc["page_content"] for doc in documents]
        if not self.embedding_client:
            logger.error("Embedding client not configured in DocumentService.")
            raise ValueError("Embedding client not configured.")
        logger.info(f"Generating embeddings for {len(texts)} document chunks.")
        try: embeddings = self.embedding_client.generate_embeddings(texts=texts)
        except Exception as e: logger.error(f"Embedding generation failed: {e}", exc_info=True); raise RuntimeError("Embedding failed.") from e
        if len(embeddings) != len(documents):
            logger.error(f"Mismatch between number of documents ({len(documents)}) and generated embeddings ({len(embeddings)}).")
            raise RuntimeError("Embedding count mismatch.")
        logger.info(f"Adding {len(documents)} documents to vector store.")
        try:
            doc_ids = self.vector_store.add_documents(documents=documents, embeddings=embeddings)
            logger.info(f"Successfully added documents to store. IDs: {doc_ids}"); return doc_ids
        except Exception as e: logger.error(f"Failed to add documents to vector store: {e}", exc_info=True); raise RuntimeError("Failed to add to store.") from e

    def retrieve_relevant_documents(self, query: str, top_k: Optional[int]=None, filters: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
        if not self.embedding_client:
            logger.error("Embedding client not configured in DocumentService for retrieval.")
            raise ValueError("Embedding client required for document retrieval.")
        logger.info(f"Generating embedding for query: '{query[:50]}...'")
        try: query_embedding = self.embedding_client.generate_embeddings(texts=[query])[0]
        except Exception as e: logger.error(f"Query embedding generation failed: {e}", exc_info=True); raise RuntimeError("Query embedding failed.") from e

        effective_top_k = top_k if top_k is not None else settings.DEFAULT_TOP_K_RETRIEVAL
        logger.info(f"Searching vector store with top_k={effective_top_k} and filters={filters}")
        try:
            results = self.vector_store.search(query_embedding=query_embedding, top_k=effective_top_k, filter_criteria=filters)
            logger.info(f"Retrieved {len(results)} documents from vector store."); return results
        except Exception as e: logger.error(f"Vector store search failed: {e}", exc_info=True); raise RuntimeError("Store search failed.") from e

    def ingest_file(self, uploaded_file: IO[bytes], file_name: str, chunker_type: str="recursive", chunker_params: Optional[Dict[str,Any]]=None, metadata_override: Optional[Dict[str,Any]]=None) -> List[str]:
        logger.info(f"Starting ingestion process for file: {file_name} with chunker: {chunker_type}")
        processed_docs = self.process_uploaded_file(uploaded_file, file_name, chunker_type, chunker_params, metadata_override)
        if not processed_docs:
            logger.warning(f"No documents were processed from file {file_name}, nothing to add to store.")
            return []
        logger.info(f"Successfully processed {len(processed_docs)} documents from {file_name}.")
        return self.add_documents_to_store(documents=processed_docs)
