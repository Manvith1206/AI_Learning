"""
Configuration manager for AI services.
This module provides utilities for managing API keys and service configurations.
"""
import os
import json
from typing import Dict, Any, Optional
import streamlit as st


class ConfigManager:
    """
    Configuration manager for AI services.
    Handles API keys and service-specific configurations.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file (optional)
        """
        self.config_file = config_file
        self.config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or environment variables."""
        # Try to load from config file
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        # Load from Streamlit secrets if available
        if hasattr(st, 'secrets'):
            for key, value in st.secrets.items():
                if key.endswith('_API_KEY'):
                    service = key.replace('_API_KEY', '').lower()
                    if service not in self.config:
                        self.config[service] = {}
                    self.config[service]['api_key'] = value
    
    def save_config(self) -> None:
        """Save configuration to file."""
        if self.config_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Save config without API keys for security
            safe_config = {}
            for service, settings in self.config.items():
                safe_config[service] = {k: v for k, v in settings.items() if k != 'api_key'}
            
            with open(self.config_file, 'w') as f:
                json.dump(safe_config, f, indent=2)
    
    def get_api_key(self, service: str):
        """
        Get the API key for a service.
        
        Args:
            service: The service name
            
        Returns:
            The API key for the service, or None if not found
        """
        if service in self.config and 'api_key' in self.config[service]:
            return self.config[service]['api_key']
        
        # Try to get from environment variables
        env_var = f"{service.upper()}_API_KEY"
        return os.environ.get(env_var)
    
    def set_api_key(self, service: str, api_key: str) -> None:
        """
        Set the API key for a service.
        
        Args:
            service: The service name
            api_key: The API key
        """
        if service not in self.config:
            self.config[service] = {}
        
        self.config[service]['api_key'] = api_key
    
    def get_service_config(self, service: str) -> Dict[str, Any]:
        """
        Get the configuration for a service.
        
        Args:
            service: The service name
            
        Returns:
            The configuration for the service
        """
        return self.config.get(service, {})
    
    def set_service_config(self, service: str, config: Dict[str, Any]) -> None:
        """
        Set the configuration for a service.
        
        Args:
            service: The service name
            config: The configuration
        """
        self.config[service] = config
