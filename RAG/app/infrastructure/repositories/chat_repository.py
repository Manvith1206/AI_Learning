from typing import Dict, List, Any, Optional
import json
import os
import uuid
from datetime import datetime
from app.core.interfaces.interfaces import ChatRepository
from app.domain.models.models import ChatSession, Message


class FileSystemChatRepository(ChatRepository):
    """File system implementation of chat repository"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file system chat repository
        
        Args:
            config: Configuration dictionary with storage_path, etc.
        """
        self.config = config
        self.storage_path = config.get("storage_path", "./chats")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Index of sessions
        self.index_path = os.path.join(self.storage_path, "index.json")
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load session index from file"""
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save session index to file"""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)
    
    def _session_to_dict(self, session: ChatSession) -> Dict[str, Any]:
        """Convert session to dictionary for storage"""
        return {
            "id": session.id,
            "created_at": session.created_at.isoformat(),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "session_id": msg.session_id
                }
                for msg in session.messages
            ]
        }
    
    def _dict_to_session(self, data: Dict[str, Any]) -> ChatSession:
        """Convert dictionary to session"""
        messages = [
            Message(
                id=msg["id"],
                role=msg["role"],
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                session_id=msg["session_id"]
            )
            for msg in data.get("messages", [])
        ]
        
        return ChatSession(
            id=data["id"],
            messages=messages,
            created_at=datetime.fromisoformat(data["created_at"])
        )
    
    def add(self, session: ChatSession) -> str:
        """
        Add a chat session to the repository
        
        Args:
            session: Chat session to add
        
        Returns:
            ID of the added session
        """
        # Generate ID if not present
        session_id = session.id if session.id else str(uuid.uuid4())
        session.id = session_id
        
        # Ensure all messages have the correct session ID
        for msg in session.messages:
            msg.session_id = session_id
        
        # Save session data
        session_path = os.path.join(self.storage_path, f"{session_id}.json")
        with open(session_path, "w") as f:
            json.dump(self._session_to_dict(session), f, indent=2)
        
        # Update index
        self.index[session_id] = {
            "id": session_id,
            "created_at": session.created_at.isoformat(),
            "message_count": len(session.messages),
            "file_path": session_path
        }
        
        self._save_index()
        return session_id
    
    def get(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID
        
        Args:
            session_id: Session ID
        
        Returns:
            Chat session if found, None otherwise
        """
        if session_id not in self.index:
            return None
        
        session_info = self.index[session_id]
        session_path = session_info["file_path"]
        
        if not os.path.exists(session_path):
            return None
        
        with open(session_path, "r") as f:
            data = json.load(f)
        
        return self._dict_to_session(data)
    
    def get_all(self) -> List[ChatSession]:
        """
        Get all chat sessions
        
        Returns:
            List of all chat sessions
        """
        sessions = []
        
        for session_id in self.index:
            session = self.get(session_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a chat session by ID
        
        Args:
            session_id: Session ID
        
        Returns:
            True if session was deleted, False otherwise
        """
        if session_id not in self.index:
            return False
        
        session_info = self.index[session_id]
        session_path = session_info["file_path"]
        
        # Delete file if it exists
        if os.path.exists(session_path):
            os.remove(session_path)
        
        # Remove from index
        del self.index[session_id]
        self._save_index()
        
        return True
    
    def update(self, session: ChatSession) -> bool:
        """
        Update a chat session
        
        Args:
            session: Chat session to update
        
        Returns:
            True if session was updated, False otherwise
        """
        if not session.id or session.id not in self.index:
            return False
        
        # Update session data
        session_path = os.path.join(self.storage_path, f"{session.id}.json")
        with open(session_path, "w") as f:
            json.dump(self._session_to_dict(session), f, indent=2)
        
        # Update index
        self.index[session.id].update({
            "message_count": len(session.messages)
        })
        
        self._save_index()
        return True
    
    def add_message(self, session_id: str, message: Message) -> bool:
        """
        Add a message to a chat session
        
        Args:
            session_id: Session ID
            message: Message to add
        
        Returns:
            True if message was added, False otherwise
        """
        session = self.get(session_id)
        if not session:
            return False
        
        # Set session ID on message
        message.session_id = session_id
        
        # Add message to session
        session.messages.append(message)
        
        # Update session
        return self.update(session)
    
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
