"""
Gemini service implementation.
This module provides an implementation of the AIService interface for Google's Gemini.
"""
import json
import inspect
from typing import List, Dict, Any, Optional, Union, Callable

# Import Google Generative AI library
from google import genai
from google.genai import types

from .base_service import AIService
from .schema_utils import get_function_properties


class GeminiService(AIService):
    """
    Google Gemini service implementation.
    """
    
    def __init__(self):
        """Initialize the Gemini service."""
        self.client = None
        
    def initialize(self, api_key: str, **kwargs):
        """
        Initialize the Gemini client with the provided API key.
        
        Args:
            api_key: The Google API key
            **kwargs: Additional parameters for the Gemini client
        """
        self.client = genai.Client(api_key=api_key)
    
    def get_function_schema(self, function: Callable):
        """
        Convert a Python function to Gemini's function schema format.
        
        Args:
            function: The Python function to convert
            
        Returns:
            Dict containing the function schema in Gemini's format
        """
        signature = inspect.signature(function)
        params = []
        for sig in signature.parameters.values():
            params.append(sig.name)
            
        properties = get_function_properties(function.__name__)
        
        # Gemini uses a different schema format with uppercase types
        return {
            "name": function.__name__,
            "description": function.__doc__,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": params
            }
        }
    
    def _convert_properties_to_gemini_format(self, properties: Dict[str, Any]):
        """
        Convert properties from the common format to Gemini's format.
        
        Args:
            properties: Properties in the common format
            
        Returns:
            Properties in Gemini's format
        """
        gemini_properties = {}
        
        for name, prop in properties.items():
            gemini_prop = prop.copy()
            
            # Convert type names to Gemini format (uppercase)
            if "type" in gemini_prop:
                if gemini_prop["type"] == "string":
                    gemini_prop["type"] = "STRING"
                elif gemini_prop["type"] == "number":
                    gemini_prop["type"] = "NUMBER"
                elif gemini_prop["type"] == "integer":
                    gemini_prop["type"] = "INTEGER"
                elif gemini_prop["type"] == "boolean":
                    gemini_prop["type"] = "BOOLEAN"
                elif gemini_prop["type"] == "array":
                    gemini_prop["type"] = "ARRAY"
                    if "items" in gemini_prop:
                        if "type" in gemini_prop["items"]:
                            gemini_prop["items"]["type"] = gemini_prop["items"]["type"].upper()
                elif gemini_prop["type"] == "object":
                    gemini_prop["type"] = "OBJECT"
                    if "properties" in gemini_prop:
                        gemini_prop["properties"] = self._convert_properties_to_gemini_format(gemini_prop["properties"])
            
            gemini_properties[name] = gemini_prop
            
        return gemini_properties
    
    def call_with_functions(
        self, 
        messages: List[Dict[str, Any]], 
        functions: List[Dict[str, Any]], 
        model: str = "gemini-1.5-pro",
        **kwargs
    ):
        """
        Call Gemini with function calling capabilities.
        
        Args:
            messages: List of message objects in the conversation
            functions: List of function schemas
            model: The model to use for the request (default: gemini-1.5-pro)
            **kwargs: Additional parameters for the Gemini API
            
        Returns:
            Response from Gemini
        """
        if not self.client:
            raise ValueError("Gemini client not initialized. Call initialize() first.")
            
        print('Messages', messages)
        # Convert messages to Gemini format
        gemini_contents = self._convert_messages_to_gemini_format(messages)
        
        # Convert functions to Gemini tool format
        tools = [types.Tool(function_declarations=[func]) for func in functions]
        
        # Create generation config
        config = types.GenerateContentConfig(
            tools=tools,
            temperature=kwargs.get("temperature", 0.7)
        )
        
        # Create model and generate content
        response = self.client.models.generate_content(
            model=model,
            contents=gemini_contents,
            config=config
        )
        
        return response
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, Any]]):
        """
        Convert messages from the common format to Gemini's format.
        
        Args:
            messages: List of message objects in the common format
            
        Returns:
            List of message objects in Gemini's format
        """
        gemini_contents = []
        system_message = None
        
        # Extract system message if present
        for message in messages:
            role = message['role'] or ""

            if role == "system":
                system_message = message['content']
                break
        
        for message in messages:
            role = message['role'] or ""
            if role != "":
                if role == "system":
                    # System message is handled with the first user message
                    continue
                elif role == "user":
                    # Add system message to the first user message if present
                    if system_message and not gemini_contents:
                        content = f"System instructions: {system_message}\n\nUser: {message['content']}"
                        gemini_contents.append(types.Content(
                            role="user",
                            parts=[types.Part(text=content)]
                        ))
                    else:
                        gemini_contents.append(types.Content(
                            role="user",
                            parts=[types.Part(text=message['content'])]
                        ))
                elif role == "assistant":
                    gemini_contents.append(types.Content(
                        role="assistant",
                        parts=[types.Part(text=message['content'])]
                    ))
                else:
                    gemini_contents.append(types.Content(
                        parts=[types.Part(text=message['content'])]
                    ))

            # elif message['type'] == "function_call_output" or hasattr(message, 'call_id'):
            #     # This is a function result from a previous interaction
            #     function_name = None
            #     result = None
                
            #     # Extract function name and result
            #     if hasattr(message, 'call_id'):
            #         # From our common format
            #         for prev_msg in messages:
            #             if hasattr(prev_msg, 'call_id') and prev_msg.call_id == message.call_id:
            #                 function_name = prev_msg.name
            #                 break
            #         result = message.output
            #     elif message['call_id']:
            #         # From our common format as dict
            #         for prev_msg in messages:
            #             if prev_msg['call_id'] == message['call_id']:
            #                 function_name = prev_msg['name']
            #                 break
            #         result = message.get('output')
                
            #     if function_name and result:
            #         # Create function response part
            #         function_response_part = types.Part.from_function_response(
            #             name=function_name,
            #             response={"result": str(result)}
            #         )
                    
            #         # Add function response to contents
            #         gemini_contents.append(types.Content(
            #             role="user",
            #             parts=[function_response_part]
            #         ))
            # elif hasattr(message, 'name') or message['name']:
            #     # This is a function call from the model
            #     # We'll handle this in extract_function_calls and create_message_from_function_result
            #     # Just add it as a placeholder here
            #     pass
                
        return gemini_contents
    
    def extract_function_calls(self, response: Any):
        """
        Extract function calls from the Gemini response.
        
        Args:
            response: The response from Gemini
            
        Returns:
            List of function calls extracted from the response
        """
        function_calls = []
        
        # Extract function calls from Gemini response
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            # Create a function call object similar to our common format
                            print("FunctionCall", part.function_call)
                            function_calls.append({
                                'name': part.function_call.name,
                                'arguments': json.dumps(part.function_call.args),
                                'call_id': f"gemini_function_{len(function_calls)}"
                            })
        
        return function_calls
    
    def create_message_from_function_result(self, function_call: Dict[str, Any], result: Any):
        """
        Create a message object from a function call result for Gemini.
        
        Args:
            function_call: The function call object
            result: The result of the function call
            
        Returns:
            Message object to be added to the conversation
        """
        # For Gemini, we need to format the function result in a way that matches our common format
        # but will be converted correctly in _convert_messages_to_gemini_format
        name = function_call.get('name')
        if not name and hasattr(function_call, 'name'):
            name = function_call.name
            
        call_id = function_call.get('call_id')
        if not call_id and hasattr(function_call, 'call_id'):
            call_id = function_call.call_id
        
                
        function_response_part = types.Part.from_function_response(
            name=name,
            response={"result": result},
        )
        print("GeminiService / MessagesFromResponseResult / FunctionCall: ", function_call)
        print("GeminiService / MessagesFromResponseResult / FunctionResult: ", function_response_part)
        function_response_part = types.Content(role="user", parts=[function_call])
        function_response_part = types.Content(role="user", parts=[function_response_part])

        return function_call, function_response_part
