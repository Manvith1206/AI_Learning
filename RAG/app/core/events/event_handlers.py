from app.core.events.event_bus import Event, event_bus
from typing import Dict, Any, List
import logging

# Configure logging
logger = logging.getLogger(__name__)


# Define standard events
class DocumentProcessedEvent(Event):
    """Event fired when a document has been processed"""
    def __init__(self, document_id: str, num_chunks: int, metadata: Dict[str, Any]):
        super().__init__(
            name="document_processed",
            data={
                "document_id": document_id,
                "num_chunks": num_chunks,
                "metadata": metadata
            }
        )


class QueryExecutedEvent(Event):
    """Event fired when a query has been executed"""
    def __init__(self, query: str, num_results: int, metadata: Dict[str, Any]):
        super().__init__(
            name="query_executed",
            data={
                "query": query,
                "num_results": num_results,
                "metadata": metadata
            }
        )


class FlashcardsGeneratedEvent(Event):
    """Event fired when flashcards have been generated"""
    def __init__(self, document_id: str, num_flashcards: int, metadata: Dict[str, Any]):
        super().__init__(
            name="flashcards_generated",
            data={
                "document_id": document_id,
                "num_flashcards": num_flashcards,
                "metadata": metadata
            }
        )


class ConfigurationUpdatedEvent(Event):
    """Event fired when configuration has been updated"""
    def __init__(self, component_name: str, config: Dict[str, Any]):
        super().__init__(
            name="configuration_updated",
            data={
                "component_name": component_name,
                "config": config
            }
        )


# Example event handlers
def log_document_processed(event: Event) -> None:
    """Log when a document has been processed"""
    logger.info(
        f"Document processed: {event.data['document_id']} "
        f"with {event.data['num_chunks']} chunks"
    )


def log_query_executed(event: Event) -> None:
    """Log when a query has been executed"""
    logger.info(
        f"Query executed: '{event.data['query']}' "
        f"with {event.data['num_results']} results"
    )


def log_configuration_updated(event: Event) -> None:
    """Log when configuration has been updated"""
    logger.info(
        f"Configuration updated for {event.data['component_name']}"
    )


# Register event handlers
def register_default_handlers():
    """Register default event handlers"""
    event_bus.subscribe("document_processed", log_document_processed)
    event_bus.subscribe("query_executed", log_query_executed)
    event_bus.subscribe("configuration_updated", log_configuration_updated)
