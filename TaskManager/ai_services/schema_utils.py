"""
Schema utilities for AI services.
This module provides utilities for working with function schemas across different AI services.
"""
import inspect
from typing import Dict, Any, Callable, List
import sys
import os

# Add the parent directory to sys.path to import CommonUtils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CommonUtils


def get_function_properties(function_name: str):
    """
    Get the properties for a function from the CommonUtils module.
    
    Args:
        function_name: The name of the function
        
    Returns:
        Dict containing the properties for the function
    """
    if hasattr(CommonUtils, 'function_properties_map') and function_name in CommonUtils.function_properties_map:
        return CommonUtils.function_properties_map[function_name]
    else:
        # Return empty properties if not found
        return {}


def register_function(function: Callable, properties: Dict[str, Any]):
    """
    Register a function and its properties in the CommonUtils module.
    
    Args:
        function: The function to register
        properties: The properties for the function
    """
    if not hasattr(CommonUtils, 'function_properties_map'):
        CommonUtils.function_properties_map = {}
    
    CommonUtils.function_properties_map[function.__name__] = properties


def convert_functions_to_schemas(functions: List[Callable], service_type: str = 'openai'):
    """
    Convert a list of functions to schemas for a specific AI service.
    
    Args:
        functions: List of functions to convert
        service_type: The type of AI service to convert for ('openai', 'anthropic', 'gemini')
        
    Returns:
        List of function schemas for the specified service
    """
    from .service_factory import get_service
    
    service = get_service(service_type)
    schemas = []
    
    for function in functions:
        schemas.append(service.get_function_schema(function))
    
    return schemas
