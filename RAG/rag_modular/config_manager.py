import os

class ConfigManager:
    """Manages configuration for RAG components"""
    def __init__(self, config=None):
        self.config = config or {
            "chunker": {"type": "recursive", "params": {"chunk_size": 600, "chunk_overlap": 200}},
            "embedder": {"type": "tfidf"},
            "vector_store": {"type": "sklearn", "params": {"metric": "cosine"}},
            "retriever": {"type": "similarity", "params": {"similarity_threshold": 0.0}, "top_k": 5},
            "reranker": {"type": "llm", "model": "gemini-2.0-flash"},
            "llm": {"type": "gemini", "model": "gemini-2.0-flash"},
            "evaluator": {"type": "simple"}
        }
    def get_config(self, component=None):
        if component:
            return self.config.get(component, {})
        return self.config
    def update_config(self, component, config):
        self.config[component] = config
