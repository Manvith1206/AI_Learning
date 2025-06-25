from typing import Any, Callable, Dict

# Type alias for a registry
Registry = Dict[str, Callable[..., Any]]

# Component Registries
CHUNKERS_REGISTRY: Registry = {}
EMBEDDERS_REGISTRY: Registry = {}
VECTOR_STORES_REGISTRY: Registry = {}
RETRIEVERS_REGISTRY: Registry = {}
LLM_SERVICES_REGISTRY: Registry = {}
RERANKERS_REGISTRY: Registry = {}
EVALUATORS_REGISTRY: Registry = {}

def register(registry: Registry, name: str):
    """A decorator to register a class in a given registry."""
    def decorator(cls):
        if name in registry:
            raise ValueError(f"Error: {name} is already registered in {registry}.")
        registry[name] = cls
        return cls
    return decorator
