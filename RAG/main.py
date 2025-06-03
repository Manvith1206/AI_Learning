import os
import streamlit as st
from dotenv import load_dotenv

from app.core.config.settings import Settings
from app.core.config.config_manager import ConfigManager
from app.core.di_container import DIContainer
from app.core.events.event_bus import event_bus
# from app.core.events.event_handlers import (
#     LoggingEventHandler, 
#     DocumentProcessedEventHandler,
#     QueryExecutedEventHandler,
#     FlashcardsGeneratedEventHandler,
#     ConfigurationUpdatedEventHandler
# )

from app.domain.services import RAGService

from app.application.usecases.document_processing import DocumentProcessingUseCase
from app.application.usecases.chat_generation import ChatGenerationUseCase
from app.application.usecases.flashcard_generation import FlashcardGenerationUseCase
from app.application.usecases.evaluation import EvaluationUseCase
from app.application.usecases.configuration import ConfigurationUseCase

from app.presentation.pages.main_page import MainPage


def setup_event_handlers():
    """Set up event handlers for the application"""
    # Register logging handlers for all events
    event_bus.subscribe("document_processed", LoggingEventHandler().handle)
    event_bus.subscribe("query_executed", LoggingEventHandler().handle)
    event_bus.subscribe("flashcards_generated", LoggingEventHandler().handle)
    event_bus.subscribe("configuration_updated", LoggingEventHandler().handle)
    
    # Register specific handlers
    event_bus.subscribe("document_processed", DocumentProcessedEventHandler().handle)
    event_bus.subscribe("query_executed", QueryExecutedEventHandler().handle)
    event_bus.subscribe("flashcards_generated", FlashcardsGeneratedEventHandler().handle)
    event_bus.subscribe("configuration_updated", ConfigurationUpdatedEventHandler().handle)


def setup_container(settings, config_manager):
    """Set up the dependency injection container"""
    container = DIContainer()
    
    # Register component factories
    container.register_factory("chunker", lambda config: create_chunker(config))
    container.register_factory("embedder", lambda config: create_embedder(config))
    container.register_factory("vector_store", lambda config: create_vector_store(config))
    container.register_factory("retriever", lambda config: create_retriever(config))
    container.register_factory("reranker", lambda config: create_reranker(config))
    container.register_factory("llm", lambda config: create_llm(config))
    container.register_factory("evaluator", lambda config: create_evaluator(config))
    
    # Initialize components with configurations
    for component_type in container.factories.keys():
        config = config_manager.get_config(component_type)
        container.get_or_create(component_type, config)
    
    return container


def create_chunker(config):
    """Create a chunker based on configuration"""
    chunker_type = config.get("type", "recursive")
    
    if chunker_type == "recursive":
        from app.infrastructure.chunkers.recursive_chunker import RecursiveChunker
        return RecursiveChunker(config.get("params", {}))
    elif chunker_type == "sentence":
        # This would be implemented in a real application
        from app.infrastructure.chunkers.recursive_chunker import RecursiveChunker
        return RecursiveChunker(config.get("params", {}))
    elif chunker_type == "semantic":
        # This would be implemented in a real application
        from app.infrastructure.chunkers.recursive_chunker import RecursiveChunker
        return RecursiveChunker(config.get("params", {}))
    else:
        from app.infrastructure.chunkers.recursive_chunker import RecursiveChunker
        return RecursiveChunker(config.get("params", {}))


def create_embedder(config):
    """Create an embedder based on configuration"""
    embedder_type = config.get("type", "openai")
    
    if embedder_type == "openai":
        from app.infrastructure.embedders.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(config.get("params", {}))
    elif embedder_type == "cohere":
        # This would be implemented in a real application
        from app.infrastructure.embedders.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(config.get("params", {}))
    elif embedder_type == "tfidf":
        # This would be implemented in a real application
        from app.infrastructure.embedders.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(config.get("params", {}))
    else:
        from app.infrastructure.embedders.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(config.get("params", {}))


def create_vector_store(config):
    """Create a vector store based on configuration"""
    vector_store_type = config.get("type", "memory")
    
    if vector_store_type == "chroma":
        from app.infrastructure.vector_store.chroma_store import ChromaVectorStore
        return ChromaVectorStore(config.get("params", {}))
    elif vector_store_type == "memory":
        from app.infrastructure.vector_store.memory_store import InMemoryVectorStore
        return InMemoryVectorStore(config.get("params", {}))
    else:
        from app.infrastructure.vector_store.memory_store import InMemoryVectorStore
        return InMemoryVectorStore(config.get("params", {}))


def create_retriever(config):
    """Create a retriever based on configuration"""
    # In a real application, this would create different retriever implementations
    # For now, we'll use a simple retriever that's part of the RAG service
    return config


def create_reranker(config):
    """Create a reranker based on configuration"""
    # In a real application, this would create different reranker implementations
    # For now, we'll use a simple reranker that's part of the RAG service
    return config


def create_llm(config):
    """Create an LLM client based on configuration"""
    llm_type = config.get("type", "mock")
    
    if llm_type == "openai":
        from app.infrastructure.llm.openai_client import OpenAIClient
        return OpenAIClient(config.get("params", {}))
    elif llm_type == "mock":
        from app.infrastructure.llm.mock_client import MockLLMClient
        return MockLLMClient(config.get("params", {}))
    else:
        from app.infrastructure.llm.mock_client import MockLLMClient
        return MockLLMClient(config.get("params", {}))


def create_evaluator(config):
    """Create an evaluator based on configuration"""
    # In a real application, this would create different evaluator implementations
    # For now, we'll use a simple evaluator that's part of the RAG service
    return config


def main():
    """Main entry point for the application"""
    # Load environment variables
    load_dotenv()
    
    # Initialize settings and config manager
    settings = Settings()
    config_manager = ConfigManager(settings)
    
    # Set up event handlers
    # setup_event_handlers()
    
    # Set up dependency injection container
    container = setup_container(settings, config_manager)
    
    # Create domain service
    rag_service = RAGService(
        chunker=container.get("chunker"),
        embedder=container.get("embedder"),
        vector_store=container.get("vector_store"),
        retriever_config=container.get("retriever"),
        reranker_config=container.get("reranker"),
        llm_service=container.get("llm"),
        evaluator_config=container.get("evaluator")
    )
    
    # Create application use cases
    document_processing_usecase = DocumentProcessingUseCase(rag_service)
    chat_generation_usecase = ChatGenerationUseCase(rag_service)
    flashcard_generation_usecase = FlashcardGenerationUseCase(rag_service)
    evaluation_usecase = EvaluationUseCase(rag_service)
    configuration_usecase = ConfigurationUseCase(config_manager, container)
    
    # Create and render main page
    main_page = MainPage(
        document_processing_usecase=document_processing_usecase,
        chat_generation_usecase=chat_generation_usecase,
        flashcard_generation_usecase=flashcard_generation_usecase,
        evaluation_usecase=evaluation_usecase,
        configuration_usecase=configuration_usecase
    )
    main_page.render()


if __name__ == "__main__":
    main()
