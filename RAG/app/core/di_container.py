from typing import Dict, Any, Type, Optional
from app.core.interfaces import (
    LLMService, VectorStore, Embedder, 
    Chunker, Retriever, Reranker, Evaluator,
    DocumentRepository, ChatRepository
)
from app.core.config.config_manager import ConfigManager
from app.core.exceptions import ComponentInitializationError
import importlib


class DIContainer:
    """Dependency Injection Container for managing component instances"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.instances: Dict[str, Any] = {}
        self.factories: Dict[str, Dict[str, Type]] = {
            "chunker": {},
            "embedder": {},
            "vector_store": {},
            "retriever": {},
            "reranker": {},
            "llm": {},
            "evaluator": {},
            "document_repository": {},
            "chat_repository": {}
        }
    
    def register_factory(self, component_type: str, name: str, factory_class: Type) -> None:
        """Register a factory class for a component type"""
        if component_type not in self.factories:
            raise ValueError(f"Unknown component type: {component_type}")
        
        self.factories[component_type][name] = factory_class
    
    def _get_implementation_class(self, component_type: str, implementation_type: str) -> Type:
        """Get the implementation class for a component type and implementation"""
        if component_type not in self.factories:
            raise ComponentInitializationError(f"Unknown component type: {component_type}")
        
        if implementation_type not in self.factories[component_type]:
            # Try dynamic import
            try:
                module_path = f"app.infrastructure.{component_type}.{implementation_type}"
                module = importlib.import_module(module_path)
                
                # Convention: Class name is the implementation type in CamelCase
                class_name = "".join(word.capitalize() for word in implementation_type.split("_"))
                
                if hasattr(module, class_name):
                    implementation_class = getattr(module, class_name)
                    # Register for future use
                    self.factories[component_type][implementation_type] = implementation_class
                    return implementation_class
                else:
                    raise ComponentInitializationError(
                        f"Implementation class {class_name} not found in {module_path}"
                    )
            except ImportError:
                raise ComponentInitializationError(
                    f"Implementation {implementation_type} for {component_type} not found"
                )
        
        return self.factories[component_type][implementation_type]
    
    def get_instance(self, component_type: str) -> Any:
        """Get or create an instance of a component"""
        if component_type in self.instances:
            return self.instances[component_type]
        
        config = self.config_manager.get_config(component_type)
        implementation_type = config["type"]
        params = config.get("params", {})
        
        # Get the implementation class
        implementation_class = self._get_implementation_class(component_type, implementation_type)
        
        # Create the instance
        try:
            instance = implementation_class(**params)
            self.instances[component_type] = instance
            return instance
        except Exception as e:
            raise ComponentInitializationError(
                f"Failed to initialize {component_type} with implementation {implementation_type}: {e}"
            )
    
    def get_chunker(self) -> Chunker:
        """Get the configured chunker instance"""
        return self.get_instance("chunker")
    
    def get_embedder(self) -> Embedder:
        """Get the configured embedder instance"""
        return self.get_instance("embedder")
    
    def get_vector_store(self) -> VectorStore:
        """Get the configured vector store instance"""
        return self.get_instance("vector_store")
    
    def get_retriever(self) -> Retriever:
        """Get the configured retriever instance"""
        return self.get_instance("retriever")
    
    def get_reranker(self) -> Reranker:
        """Get the configured reranker instance"""
        return self.get_instance("reranker")
    
    def get_llm_service(self) -> LLMService:
        """Get the configured LLM service instance"""
        return self.get_instance("llm")
    
    def get_evaluator(self) -> Evaluator:
        """Get the configured evaluator instance"""
        return self.get_instance("evaluator")
    
    def update_component(self, component_type: str, config: Dict[str, Any]) -> None:
        """Update a component's configuration and recreate the instance"""
        self.config_manager.update_config(component_type, config)
        
        # Remove the existing instance to force recreation
        if component_type in self.instances:
            del self.instances[component_type]
