from typing import Dict, Any, Optional
import json
import os
from copy import deepcopy

from app.core.config.settings import get_default_config


class ConfigManager:
    """Manages configuration for all RAG components"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize with default config or from file if provided"""
        self.config = get_default_config()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def get_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        if component_name not in self.config:
            raise ValueError(f"Unknown component: {component_name}")
        
        return deepcopy(self.config[component_name])
    
    def update_config(self, component_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific component"""
        if component_name not in self.config:
            raise ValueError(f"Unknown component: {component_name}")
        
        self.config[component_name] = deepcopy(config)
    
    def save_to_file(self, file_path: str) -> None:
        """Save current configuration to a file"""
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from a file"""
        with open(file_path, 'r') as f:
            loaded_config = json.load(f)
            
        # Merge with defaults to ensure all required keys exist
        for component, config in loaded_config.items():
            if component in self.config:
                self.config[component] = config
