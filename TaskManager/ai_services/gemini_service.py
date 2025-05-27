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
        Convert a list of messages from our common format to Gemini's format.
        """
        gemini_contents = []
        system_prompt_text = None

        # Extract system prompt first if it exists (Gemini prefers it separately or early)
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg:
                if msg['role'] == "system":
                    system_prompt_text = msg['content']
                    break # Assuming one system prompt
        
        # If a system prompt exists and the service instance can hold it, set it.
        # Note: google.generativeai.GenerativeModel has a system_instruction parameter.
        # This conversion focuses on the 'contents' part. Handling system_instruction 
        # would typically be done when initializing the model or a chat session.
        # For now, we'll prepend it to the first user message if not handled at a higher level.

        is_first_user_message = True
        for message in messages:
            role, content = "", ""
            tool_calls = None
            if isinstance(message, dict) and "role" in message:
                role = message["role"]
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
            if isinstance(message, dict) and "tool_calls" in message:
                tool_calls = message["tool_calls"]

            if role == "system":
                continue # Handled above or by a dedicated system_instruction mechanism

            gemini_role = "user" # Default
            parts = []

            if role == "user":
                gemini_role = "user"
                text_to_send = str(content or "")
                if system_prompt_text and is_first_user_message:
                    # Prepend system prompt to the first user message's text part
                    # This is a common workaround if not using native system_instruction
                    text_to_send = f"System Instructions:\n{system_prompt_text}\n\n{text_to_send}"
                    system_prompt_text = None # Consume it
                is_first_user_message = False
                if text_to_send:
                    parts.append(types.Part(text=text_to_send))

            elif role == "assistant":
                gemini_role = "model"
                if tool_calls:
                    for tc in tool_calls:
                        function_details = tc.get("function")
                        if function_details:
                            name = function_details.get("name")
                            try:
                                args = json.loads(function_details.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                args = {}
                            parts.append(types.Part(function_call=types.FunctionCall(name=name, args=args)))
                if content: # Assistant's textual response
                    parts.append(types.Part(text=str(content)))
            
            elif isinstance(message, dict) and "type" in message:  
                if role == "function" or message['type'] == "tool_result" or message['type'] == "function_call_output":
                    gemini_role = "tool" # Gemini uses 'tool' role for function responses
                    if isinstance(message, dict) and "name" in message:
                        function_name = message['name']
                    function_response_content = str(message['content'])
                    tool_call_id = message['tool_call_id'] or message['call_id'] # For matching if needed by API, though Part.from_function_response doesn't use id explicitly

                    # Gemini expects the 'response' in Part.from_function_response to be a dict.
                    response_data = {}
                    try:
                        # Try to parse the content if it's a JSON string representing a dict.
                        parsed_data = json.loads(function_response_content)
                        if isinstance(parsed_data, dict):
                            response_data = parsed_data
                        else:
                            response_data = {"result": parsed_data}
                    except (json.JSONDecodeError, TypeError):
                        # If not a JSON dict, wrap the raw content.
                        response_data = {"result": function_response_content}

                    parts.append(types.Part.from_function_response(
                        name=function_name,
                        response=response_data
                        # tool_call_id is not directly used by from_function_response here
                    ))
            else:
                print(f"DEBUG: Unknown role or unhandled message type for Gemini conversion: {message}")
                continue

            if parts:
                gemini_contents.append(types.Content(role=gemini_role, parts=parts))
            elif role == "user" and not parts: # Handle empty user message if necessary, Gemini might error
                parts.append(types.Part(text="")) # Send empty text for user role if no content
                gemini_contents.append(types.Content(role=gemini_role, parts=parts))
                
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
                            try:
                                # Handle args properly - could be a dict or a string
                                args = part.function_call.args
                                if isinstance(args, str):
                                    args_str = args
                                else:
                                    args_str = json.dumps(args)
                                
                                # Create a unique call_id for this function call
                                call_id = f"gemini_function_{len(function_calls)}"
                                
                                # Add to function calls list
                                function_calls.append({
                                    'name': part.function_call.name,
                                    'arguments': args_str,
                                    'call_id': call_id
                                })
                                
                                # Debug output
                                print(f"Extracted function call: {part.function_call.name} with args: {args_str}")
                            except Exception as e:
                                print(f"Error extracting function call: {e}")
        
        if not function_calls:
            print("No function calls found in Gemini response")
            
        return function_calls
    
    def create_message_from_function_result(self, function_call: Dict[str, Any], result: Any):
        """
        Create a message object from a function call result for Gemini.
        
        Args:
            function_call: The function call object
            result: The result of the function call
            
        Returns:
            Two message objects to be added to the conversation:
            1. The assistant's tool call message
            2. The tool response message
        """
        # Extract function name and call_id
        name = function_call.get('name')
        if not name and hasattr(function_call, 'name'):
            name = function_call.name
            
        call_id = function_call.get('call_id')
        if not call_id and hasattr(function_call, 'call_id'):
            call_id = function_call.call_id
        
        # Extract arguments
        raw_args = function_call.get("arguments") or getattr(function_call, "arguments", None)
        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
        
        # Create the assistant message with tool call
        assistant_message = {
            "role": "assistant",
            "content": "",  # Empty content since this is a function call
            "tool_calls": [{
                "function": {
                    "name": name,
                    "arguments": json.dumps(args)
                },
                "id": call_id
            }]
        }
        
        # Create the function result message
        function_result_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": str(result)
        }
        
        return assistant_message, function_result_message
