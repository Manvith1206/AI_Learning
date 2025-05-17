"""
Service factory for AI services.
This module provides a factory for creating AI service instances.
"""
from typing import Dict, Any, Optional

from .base_service import AIService
from .openai_service import OpenAIService
from .anthropic_service import AnthropicService
from .gemini_service import GeminiService


# Registry of available services
_SERVICE_REGISTRY = {
    'openai': OpenAIService,
    'claude': AnthropicService,
    'gemini': GeminiService
}


def register_service(service_type: str, service_class: type) -> None:
    """
    Register a new AI service type.
    
    Args:
        service_type: The type name for the service
        service_class: The class for the service
    """
    _SERVICE_REGISTRY[service_type] = service_class


def get_service(service_type: str):
    """
    Get an instance of the specified AI service.
    
    Args:
        service_type: The type of AI service to get
        
    Returns:
        An instance of the specified AI service
        
    Raises:
        ValueError: If the service type is not registered
    """
    if service_type not in _SERVICE_REGISTRY:
        raise ValueError(f"Service type '{service_type}' not registered")
    
    return _SERVICE_REGISTRY[service_type]()


def get_available_services() -> list:
    """
    Get a list of available service types.
    
    Returns:
        List of available service types
    """
    return list(_SERVICE_REGISTRY.keys())
