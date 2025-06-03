from typing import Dict, List, Any, Optional
import os
import openai
from app.core.interfaces.interfaces import LLMService
from app.domain.models.models import Message


class OpenAIClient(LLMService):
    """OpenAI LLM client implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI client
        
        Args:
            config: Configuration dictionary with model, temperature, etc.
        """
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
    
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate text based on a prompt
        
        Args:
            prompt: The prompt to generate from
            context: Optional context to include
        
        Returns:
            Generated text
        """
        messages = []
        
        if context:
            messages.append({"role": "system", "content": context})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_with_history(self, messages: List[Message], context: Optional[str] = None) -> str:
        """
        Generate text based on conversation history
        
        Args:
            messages: List of conversation messages
            context: Optional context to include
        
        Returns:
            Generated text
        """
        formatted_messages = []
        
        if context:
            formatted_messages.append({"role": "system", "content": context})
        
        for message in messages:
            formatted_messages.append({
                "role": message.role,
                "content": message.content
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the client configuration
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
        self.model = config.get("model", self.model)
        self.temperature = config.get("temperature", self.temperature)
        self.max_tokens = config.get("max_tokens", self.max_tokens)
