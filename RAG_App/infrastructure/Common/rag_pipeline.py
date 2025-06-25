import os
from typing import Dict, List
import uuid
import logging
from models import AppConfig
from infrastructure.common.cache_manager import CacheManager
from infrastructure.common.pipelines.component_manager import ComponentManager
from infrastructure.common.pipelines.document_processing import DocumentProcessing
from infrastructure.common.pipelines.flash_cards_generation import FlashCardGeneration
from infrastructure.common.pipelines.query_evaluation import QueryEvaluation
from infrastructure.common.pipelines.query_processing import QueryProcessing
from infrastructure.vector_stores.base_vector_store import BaseVectorStore
import infrastructure.common.RAG_Constants as constants

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Manages the end-to-end RAG workflow, from document processing to querying."""

    def __init__(self, config: AppConfig, vector_store: BaseVectorStore = None):
        logger.info("Initializing RAGPipeline...")
        self.config = config
        self.vector_store = vector_store
        self.cache_manager = CacheManager()

        # Build all components using the new ComponentManager
        component_manager = ComponentManager(config)
        self.components = component_manager.build_all_components()

        # If an external vector store is provided, use it.
        if self.vector_store:
            self.components[constants.CONFIG_VECTOR_STORE] = self.vector_store
        else:
            self.vector_store = self.components[constants.CONFIG_VECTOR_STORE]

        # Initialize sub-pipelines with the built components
        self._init_sub_pipelines()
        logger.info("RAGPipeline initialized successfully.")

    def _init_sub_pipelines(self):
        """Initializes the processing and querying sub-pipelines."""
        self.document_processing = DocumentProcessing(
            vector_store=self.components[constants.CONFIG_VECTOR_STORE],
            chunker=self.components[constants.CONFIG_CHUNKER],
            embedder=self.components[constants.CONFIG_EMBEDDER]
        )

        self.flashcards_generation = FlashCardGeneration(
            llm_service=self.components[constants.CONFIG_LLM]
        )

        self.query_evaluation = QueryEvaluation(
            evaluator=self.components[constants.CONFIG_EVALUATOR]
        )

        self.query_processing = QueryProcessing(
            llm_service=self.components[constants.CONFIG_LLM],
            vector_store=self.vector_store,
            embedder=self.components[constants.CONFIG_EMBEDDER],
            retriever=self.components[constants.CONFIG_RETRIEVER],
            reranker=self.components[constants.CONFIG_RERANKER]
        )

    def process_document(self, file_path: str, texts: list[str] = None):
        """Processes a document, updates the vector store, and syncs sub-pipelines."""
        logger.info(f"Processing document: {file_path}")
        
        with open(file_path, 'rb') as f:
            file_content = f.read()

        params = {
            'chunker': self.config.chunker.dict(),
            'embedder': self.config.embedder.dict(),
        }
        cache_key = self.cache_manager.generate_cache_key(file_content, params)
        cached_vector_store = self.cache_manager.load_from_cache(cache_key)

        if cached_vector_store:
            logger.info(f"Loading cached vector store for {file_path}")
            self.vector_store = cached_vector_store
            self.query_processing.update_vector_store(self.vector_store)
            return self.vector_store

        populated_vector_store = self.document_processing.process_document(file_path, texts)
        
        if populated_vector_store:
            self.vector_store = populated_vector_store
            self.cache_manager.save_to_cache(cache_key, populated_vector_store)
            # Ensure the query processing pipeline has the updated vector store
            self.query_processing.update_vector_store(self.vector_store)
            logger.info("Vector store updated and synchronized with query pipeline.")

        return populated_vector_store