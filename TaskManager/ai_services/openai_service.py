"""
OpenAI service implementation.
This module provides an implementation of the AIService interface for OpenAI.
"""
import json
import inspect
from typing import List, Dict, Any, Optional, Union, Callable
from openai import OpenAI

from .base_service import AIService
from .schema_utils import get_function_properties


class OpenAIService(AIService):
    """
    OpenAI service implementation.
    """
    
    def __init__(self):
        """Initialize the OpenAI service."""
        self.client = None
        
    def initialize(self, api_key: str, **kwargs):
        """
        Initialize the OpenAI client with the provided API key.
        
        Args:
            api_key: The OpenAI API key
            **kwargs: Additional parameters for the OpenAI client
        """
        self.client = OpenAI(api_key=api_key, **kwargs)
    
    def get_function_schema(self, function: Callable):
        """
        Convert a Python function to OpenAI's function schema format.
        
        Args:
            function: The Python function to convert
            
        Returns:
            Dict containing the function schema in OpenAI's format
        """
        signature = inspect.signature(function)
        params = []
        for sig in signature.parameters.values():
            params.append(sig.name)
            
        properties = get_function_properties(function.__name__)
        
        return {
            "type": "function",
            "name": function.__name__,
            "description": function.__doc__,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": params
            }
        }
    
    def call_with_functions(
        self, 
        messages: List[Dict[str, Any]], 
        functions: List[Dict[str, Any]], 
        model: str = "gpt-4-turbo",
        **kwargs
    ):
        """
        Call OpenAI with function calling capabilities.
        
        Args:
            messages: List of message objects in the conversation
            functions: List of function schemas
            model: The model to use for the request (default: gpt-4-turbo)
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            Response from OpenAI
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Call initialize() first.")
            
        # For OpenAI's newer models that use responses.create
        if hasattr(self.client, 'responses'):
            
            response = self.client.responses.create(
                model="gpt-4-1106-preview",
                input=messages,
                tools=functions,
                tool_choice=kwargs.get('tool_choice', 'auto')
            )
            print("Response", response.output)
            
            return response
        # Fallback for older models that use chat.completions.create
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                functions=functions,
                function_call=kwargs.get('function_call', 'auto')
            )
            return response
    
    def extract_function_calls(self, response: Any):
        """
        Extract function calls from the OpenAI response.
        
        Args:
            response: The response from OpenAI
            
        Returns:
            List of function calls extracted from the response
        """
        tool_calls = []
        
        # Handle newer response format (responses.create)
        if hasattr(response, 'output'):
            for resp in response.output:
                if resp.type == "function_call":
                    tool_calls.append(resp)
        # Handle older response format (chat.completions.create)
        elif hasattr(response, 'choices'):
            message = response.choices[0].message
            if hasattr(message, 'function_call') and message.function_call:
                # Convert to the same format as tool calls for consistency
                tool_calls.append({
                    'name': message.function_call.name,
                    'arguments': message.function_call.arguments,
                    'call_id': 'function_call_1'  # Assign a default ID
                })
                
        return tool_calls
    
    def create_message_from_function_result(self, function_call: Dict[str, Any], result: Any):
        """
        Create a message object from a function call result for OpenAI.
        
        Args:
            function_call: The function call object
            result: The result of the function call
            
        Returns:
            Message object to be added to the conversation
        """
        
        # For newer response format
        if hasattr(function_call, 'call_id'):
            function_call_output = {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": str(result)
            }
            return function_call,function_call_output
        
        # For older response format
        else:
            return function_call, {
                "role": "function",
                "name": function_call.get('name'),
                "content": str(result)
            }
