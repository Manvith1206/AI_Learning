"""
Base interface for AI services.
This module defines the abstract base class that all AI service implementations must follow.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable


class AIService(ABC):
    """
    Abstract base class for AI services.
    All AI service implementations (OpenAI, Anthropic, Gemini, etc.) must implement this interface.
    """
    
    @abstractmethod
    def initialize(self, api_key: str, **kwargs):
        """
        Initialize the AI service with the required credentials.
        
        Args:
            api_key: The API key for the service
            **kwargs: Additional service-specific parameters
        """
        pass
    
    @abstractmethod
    def get_function_schema(self, function: Callable):
        """
        Convert a Python function to the service-specific function schema format.
        
        Args:
            function: The Python function to convert
            
        Returns:
            Dict containing the function schema in the service's format
        """
        pass
    
    @abstractmethod
    def call_with_functions(
        self, 
        messages: List[Dict[str, Any]], 
        functions: List[Dict[str, Any]], 
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the AI service with function calling capabilities.
        
        Args:
            messages: List of message objects in the conversation
            functions: List of function schemas
            model: The model to use for the request
            **kwargs: Additional service-specific parameters
            
        Returns:
            Response from the AI service
        """
        pass
    
    @abstractmethod
    def extract_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract function calls from the AI service response.
        
        Args:
            response: The response from the AI service
            
        Returns:
            List of function calls extracted from the response
        """
        pass
    
    @abstractmethod
    def create_message_from_function_result(self, function_call: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """
        Create a message object from a function call result.
        
        Args:
            function_call: The function call object
            result: The result of the function call
            
        Returns:
            Message object to be added to the conversation
        """
        pass
