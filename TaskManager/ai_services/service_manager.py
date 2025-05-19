"""
Service manager for AI services.
This module provides a high-level interface for working with AI services.
"""
from typing import List, Dict, Any, Optional, Union, Callable
import inspect
import sys
import os

from .base_service import AIService
from .service_factory import get_service
from .config_manager import ConfigManager
from .schema_utils import convert_functions_to_schemas


class ServiceManager:
    """
    Service manager for AI services.
    Provides a high-level interface for working with different AI services.
    """
    
    def __init__(self, service_type: str = 'openai', config_file: Optional[str] = None):
        """
        Initialize the service manager.
        
        Args:
            service_type: The type of AI service to use (default: openai)
            config_file: Path to the configuration file (optional)
        """
        self.service_type = service_type
        self.service = get_service(service_type)
        self.config_manager = ConfigManager(config_file)
        
        # Initialize the service with API key
        api_key = self.config_manager.get_api_key(service_type)
        if api_key:
            self.service.initialize(api_key)
    
    def change_service(self, service_type: str):
        """
        Change the AI service being used.
        
        Args:
            service_type: The type of AI service to use
        """
        self.service_type = service_type

        self.service = get_service(service_type)
        # Initialize the service with API key
        api_key = self.config_manager.get_api_key(service_type)
        if api_key:
            self.service.initialize(api_key)
    
    def register_functions(self, functions: List[Callable]):
        """
        Register functions for function calling.
        
        Args:
            functions: List of functions to register
        """
        # Add the parent directory to sys.path to import CommonUtils
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import CommonUtils
        
        # Register functions in CommonUtils
        CommonUtils.function_schemas = []
        for function in functions:
            schema = self.service.get_function_schema(function)
            CommonUtils.function_schemas.append(schema)
            
            # Create function map if it doesn't exist
            if not hasattr(CommonUtils, 'function_map'):
                CommonUtils.function_map = {}
            
            # Add function to function map
            CommonUtils.function_map[function.__name__] = function
    
    def call_llm(self, user_input: str, messages: List[Dict[str, Any]], model: None):
        """
        Call the AI service with function calling capabilities.
        
        Args:
            user_input: The user input
            messages: List of message objects in the conversation
            model: The model to use for the request (optional)
            
        Returns:
            Response from the AI service and processed messages
        """
        # Add the parent directory to sys.path to import CommonUtils
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import CommonUtils
        
        # Add user input to messages
        messages.append({"role": "user", "content": user_input})
        
        # Use default model if not specified
        if model is None:
            model = self._get_default_model()
        
        # Call the AI service
        function_schemas = []
        functions = []
        for function in CommonUtils.function_map:
            functions.append(CommonUtils.function_map[function])
        function_schemas = convert_functions_to_schemas(functions, self.service_type) 

        response = self.service.call_with_functions(
            messages=messages,
            functions=function_schemas,
            model=model
        )
        # Extract function calls
        tool_calls = self.service.extract_function_calls(response)
        
        # Handle function calls
        for tool_call in tool_calls:
            self._handle_tool_call(tool_call, messages)
        
        # Get final response if there were function calls
        if tool_calls:
            print("ServiceManager / Messages: ", messages)
            final_response = self.service.call_with_functions(
                messages=messages,
                functions=[],
                model=model
            )
            print("Final Response: ", final_response)
            # Add assistant response to messages
            if hasattr(final_response, 'output_text'):
                messages.append({"role": "assistant", "content": final_response.output_text})
            elif hasattr(final_response, 'content'):
                if (final_response.content[0].type == 'text'):
                    messages.append({"role": "assistant", "content": final_response.content[0].text})
            elif hasattr(final_response, 'choices'):
                messages.append({"role": "assistant", "content": final_response.choices[0].message.content})
            # Handle Gemini response format
            elif hasattr(final_response, 'candidates') and final_response.candidates:
                for candidate in final_response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                messages.append({"role": "assistant", "content": part.text})
                                break
        else:
            final_response = response
            
            # Add assistant response to messages
            if hasattr(final_response, 'output_text'):
                messages.append({"role": "assistant", "content": final_response.output_text})
            elif hasattr(final_response, 'choices'):
                messages.append({"role": "assistant", "content": final_response.choices[0].message.content})
            # Handle Gemini response format
            elif hasattr(final_response, 'candidates') and final_response.candidates:
                for candidate in final_response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                messages.append({"role": "assistant", "content": part.text})
                                break
        
        return {
            "response": final_response,
            "messages": messages
        }
    
    def _handle_tool_call(self, tool_call: Dict[str, Any], messages: List[Dict[str, Any]]):
        """
        Handle a function call from the AI service.
        
        Args:
            tool_call: The function call object
            messages: List of message objects in the conversation
        """
        # Add the parent directory to sys.path to import CommonUtils
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import CommonUtils
        import json
        
        # Get function name and arguments
        if hasattr(tool_call, 'name'):
            fn_name = tool_call.name
            args = json.loads(tool_call.arguments)
        else:
            fn_name = tool_call.get('name')
            args = json.loads(tool_call.get('arguments', '{}'))
        
        # Call the function
        result = CommonUtils.function_map[fn_name](**args)
                
        # Add function result to messages
        tool_use_msg, result_message = self.service.create_message_from_function_result(tool_call, result)
        messages.append(tool_use_msg)
        messages.append(result_message)
        print("Tooluse", tool_use_msg)
        print("resultmsg", result_message)
    
    def _get_default_model(self):
        """
        Get the default model for the current service.
        
        Returns:
            The default model name
        """
        if self.service_type == 'openai':
            return "gpt-4-turbo"
        elif self.service_type == 'claude':
            return "claude-3-sonnet-20240229"
        elif self.service_type == 'gemini':
            return "gemini-2.0-flash"
        else:
            return "default"
