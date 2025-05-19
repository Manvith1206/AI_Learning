"""
Anthropic service implementation.
This module provides an implementation of the AIService interface for Anthropic.
"""
import json
import inspect
from typing import List, Dict, Any, Optional, Union, Callable

# Import will be required when implementing
from anthropic import Anthropic

from .base_service import AIService
from .schema_utils import get_function_properties


class AnthropicService(AIService):
    """
    Anthropic service implementation.
    """
    
    def __init__(self):
        """Initialize the Anthropic service."""
        self.client = None
        
    def initialize(self, api_key: str, **kwargs):
        """
        Initialize the Anthropic client with the provided API key.
        
        Args:
            api_key: The Anthropic API key
            **kwargs: Additional parameters for the Anthropic client
        """
        self.client = Anthropic(api_key=api_key, **kwargs)
    
    def get_function_schema(self, function: Callable):
        """
        Convert a Python function to Anthropic's tool schema format.
        
        Args:
            function: The Python function to convert
            
        Returns:
            Dict containing the function schema in Anthropic's format
        """
        signature = inspect.signature(function)
        params = []
        for sig in signature.parameters.values():
            params.append(sig.name)
            
        properties = get_function_properties(function.__name__)
        
        # Anthropic uses a different schema format with input_schema
        return {
            "name": function.__name__,
            "description": function.__doc__,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": params
            }
        }
    
    def call_with_functions(
        self, 
        messages: List[Dict[str, Any]], 
        functions: List[Dict[str, Any]], 
        model: str = "claude-3-sonnet-20240229",
        **kwargs
    ):
        """
        Call Anthropic with function calling capabilities.
        
        Args:
            messages: List of message objects in the conversation
            functions: List of function schemas (tools for Anthropic)
            model: The model to use for the request (default: claude-3-sonnet-20240229)
            **kwargs: Additional parameters for the Anthropic API
            
        Returns:
            Response from Anthropic
        """
        if not self.client:
            raise ValueError("Anthropic client not initialized. Call initialize() first.")
            
        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic_format(messages)
        # Anthropic uses 'tools' instead of 'functions'
        response = self.client.messages.create(
            model=model,
            messages=anthropic_messages,
            tools=functions,
            max_tokens=kwargs.get('max_tokens', 1024)
        )
        
        return response
    
    def _convert_messages_to_anthropic_format(self, messages: List[Dict[str, Any]]):
        """
        Convert messages from the common format to Anthropic's format.
        
        Args:
            messages: List of message objects in the common format
            
        Returns:
            List of message objects in Anthropic's format
        """
        anthropic_messages = []
        system_message = None
        
        # Extract system message if present
        for message in messages:
            if message.get("role") == "system":
                system_message = message.get("content")
                break
        
        for message in messages:
            role = message.get("role") or ""
            if role != "":
                if role == "system":
                    # System message is handled separately in Anthropic
                    continue
                elif role == "user":
                    # Add system message to the first user message if present
                    if system_message and not anthropic_messages:
                        anthropic_messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": f"<system>\n{system_message}\n</system>\n\n{message.get('content')}"}]
                        })
                    else:
                        anthropic_messages.append({
                            "role": "user",
                            "content": message.get("content")
                        })
                else:
                    anthropic_messages.append(message)
        
        return anthropic_messages
    
    def extract_function_calls(self, response: Any):
        """
        Extract function calls from the Anthropic response.
        
        Args:
            response: The response from Anthropic
            
        Returns:
            List of function calls extracted from the response
        """
        tool_calls = []
        # Extract tool calls from Anthropic response
        if hasattr(response, 'content'):
            for content_item in response.content:
                if hasattr(content_item, 'type') and content_item.type == "tool_use":
                    # Create a tool call object similar to our common format
                    tool_calls.append({
                        'name': content_item.name,
                        'arguments': json.dumps(content_item.input),
                        'call_id': content_item.id,
                        'id': content_item.id  # For compatibility
                    })
        
        return tool_calls
    
    def create_message_from_function_result(self, function_call: Dict[str, Any], result: Any):
        """
        Create a message object from a function call result for Anthropic.
        
        Args:
            function_call: The function call object
            result: The result of the function call
            
        Returns:
            Message object to be added to the conversation
        """
        call_id = function_call.get('call_id') or function_call.get('id')
        if not call_id and hasattr(function_call, 'id'):
            call_id = function_call.id
        elif not call_id and hasattr(function_call, 'call_id'):
            call_id = function_call.call_id
        
        args = function_call.get('arguments')
        dict_args_obj = json.loads(args)

        tool_use_msg = {     
            "role": "assistant",
            "content":[{
                "type": "tool_use",
                "id": call_id,
                "input": dict_args_obj,
                "name": function_call.get('name'),
            }   ]
        }
        tool_result_msg = {     
            "role": "user",
            "content":[{
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": str(result)
            }   ]
        }

        return tool_use_msg, tool_result_msg

