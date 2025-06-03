from typing import Dict, List, Any, Optional
from app.core.config.config_manager import ConfigManager
from app.core.di_container import DIContainer
from app.core.events.event_bus import event_bus
from app.core.events.event_handlers import ConfigurationUpdatedEvent


class ConfigurationUseCase:
    """Application use case for managing component configurations"""
    
    def __init__(self, config_manager: ConfigManager, di_container: DIContainer):
        self.config_manager = config_manager
        self.di_container = di_container
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        return self.config_manager.get_config(component_name)
    
    def update_component_config(self, component_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific component"""
        # Update config
        self.config_manager.update_config(component_name, config)
        
        # Update component instance
        self.di_container.update_component(component_name, config)
        
        # Publish event
        event_bus.publish(ConfigurationUpdatedEvent(
            component_name=component_name,
            config=config
        ))
    
    def save_config_to_file(self, file_path: str) -> None:
        """Save current configuration to a file"""
        self.config_manager.save_to_file(file_path)
    
    def load_config_from_file(self, file_path: str) -> None:
        """Load configuration from a file"""
        self.config_manager.load_from_file(file_path)
        
        # Refresh all components
        for component_type in self.di_container.factories.keys():
            if component_type in self.di_container.instances:
                del self.di_container.instances[component_type]
