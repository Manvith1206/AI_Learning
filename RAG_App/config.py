import yaml
from pydantic import ValidationError
from models import AppConfig
import streamlit as st

class ConfigManager:
    """Manages RAG component configurations by loading from a YAML file."""
    _config: AppConfig = None
    _config_path: str = "config.yml"

    @classmethod
    def load_config(cls) -> AppConfig:
        """Loads, validates, and returns the application configuration.
        Caches the configuration after the first load.
        """
        if cls._config is None:
            try:
                with open(cls._config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                cls._config = AppConfig(**config_data)
            except FileNotFoundError:
                st.error(f"Configuration file not found at: {cls._config_path}")
                raise
            except ValidationError as e:
                st.error(f"Configuration validation error in '{cls._config_path}':\n{e}")
                raise
            except Exception as e:
                st.error(f"An unexpected error occurred while loading the configuration: {e}")
                raise
        return cls._config

    @classmethod
    def get_config(cls) -> AppConfig:
        """Returns the loaded application configuration."""
        return cls.load_config()

    @classmethod
    def get_component_config(cls, component_name: str):
        """Gets the configuration for a specific component by its name."""
        config = cls.get_config()
        if hasattr(config, component_name):
            return getattr(config, component_name)
        raise AttributeError(f"Component '{component_name}' not found in configuration.")
