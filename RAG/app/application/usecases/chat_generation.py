from typing import Dict, List, Any, Optional
from app.domain.models import QueryResult, Message, ChatSession
from app.domain.services import RAGService
from app.core.events.event_bus import event_bus
from app.core.events.event_handlers import QueryExecutedEvent
import uuid
from datetime import datetime


class ChatGenerationUseCase:
    """Application use case for chat and query generation"""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.current_session: Optional[ChatSession] = None
    
    def create_session(self) -> ChatSession:
        """Create a new chat session"""
        self.current_session = ChatSession(
            id=str(uuid.uuid4()),
            messages=[],
            created_at=datetime.now()
        )
        return self.current_session
    
    def get_or_create_session(self) -> ChatSession:
        """Get current session or create a new one if none exists"""
        if not self.current_session:
            return self.create_session()
        return self.current_session
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the current session"""
        session = self.get_or_create_session()
        
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            session_id=session.id
        )
        
        session.messages.append(message)
        return message
    
    def generate_response(self, query: str) -> QueryResult:
        """Generate a response to a user query"""
        # Add user message to session
        self.add_message("user", query)
        
        # Generate response using RAG service
        result = self.rag_service.query(query)
        
        # Add assistant message to session
        self.add_message("assistant", result.answer)
        
        # Publish event
        event_bus.publish(QueryExecutedEvent(
            query=query,
            num_results=len(result.retrieved_documents),
            metadata={"rerank_explanation": result.rerank_explanation}
        ))
        
        return result
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the current chat history"""
        session = self.get_or_create_session()
        
        return [
            {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp.isoformat()
            }
            for message in session.messages
        ]
    
    def clear_chat_history(self) -> None:
        """Clear the current chat history"""
        if self.current_session:
            self.current_session.messages = []
