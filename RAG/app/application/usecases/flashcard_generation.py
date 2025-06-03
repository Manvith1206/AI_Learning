from typing import Dict, List, Any, Optional
from app.domain.models import Document, Flashcard
from app.domain.services import RAGService
from app.core.events.event_bus import event_bus
from app.core.events.event_handlers import FlashcardsGeneratedEvent


class FlashcardGenerationUseCase:
    """Application use case for flashcard generation from documents"""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.flashcards: List[Flashcard] = []
    
    def generate_flashcards(self, document: Document, num_cards: int = 5) -> List[Flashcard]:
        """Generate flashcards from a document"""
        # Generate flashcards using RAG service
        self.flashcards = self.rag_service.generate_flashcards(document, num_cards)
        
        # Publish event
        event_bus.publish(FlashcardsGeneratedEvent(
            document_id=document.id,
            num_flashcards=len(self.flashcards),
            metadata={"document_name": document.metadata.get("filename", "")}
        ))
        
        return self.flashcards
    
    def get_flashcards(self) -> List[Flashcard]:
        """Get all generated flashcards"""
        return self.flashcards
    
    def clear_flashcards(self) -> None:
        """Clear all generated flashcards"""
        self.flashcards = []
