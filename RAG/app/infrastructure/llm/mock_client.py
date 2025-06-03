from typing import Dict, List, Any, Optional
from app.core.interfaces.interfaces import LLMService
from app.domain.models.models import Message


class MockLLMClient(LLMService):
    """Mock LLM client for testing and development"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mock LLM client
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.response_template = config.get("response_template", 
            "This is a mock response to: '{prompt}'. Context: {context}"
        )
    
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate mock text based on a prompt
        
        Args:
            prompt: The prompt to generate from
            context: Optional context to include
        
        Returns:
            Generated mock text
        """
        context_str = context if context else "No context provided"
        return self.response_template.format(prompt=prompt, context=context_str)
    
    def generate_with_history(self, messages: List[Message], context: Optional[str] = None) -> str:
        """
        Generate mock text based on conversation history
        
        Args:
            messages: List of conversation messages
            context: Optional context to include
        
        Returns:
            Generated mock text
        """
        last_message = messages[-1].content if messages else "No messages"
        context_str = context if context else "No context provided"
        
        return f"Mock response to conversation. Last message: '{last_message}'. Context: {context_str}"
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the client configuration
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
        self.response_template = config.get("response_template", self.response_template)
