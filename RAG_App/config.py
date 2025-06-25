from infrastructure.common.rag_constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType, GeminiLLMModel
)

import infrastructure.common.rag_constants as constants

class ConfigManager:
    """Manages configuration for RAG components"""
    def __init__(self, config=None):
        self.config = config or   {
    "chunker": {
      "type": "recursive",
      "params": {
        "chunk_size": 150,
        "chunk_overlap": 70
      }
    },
    "embedder": {
      "type": "Voyage",
      "params": {
        "model": "voyage-3-lite"
      }
    },
    "vector_store": {
      "type": "faiss",
      "params": {}
    },
    "retriever": {
      "type": "similarity",
      "params": {
        "similarity_threshold": 0.0,
        "top_k": 20
      }
    },
    "reranker": {
      "type": "Jina",
      "params": {
        "top_k_for_reranking": 5,
        "model": "jina-reranker-v1-turbo-en"
      }
    },
    "llm": {
      "type": "Gemini",
      "params": {
        "model": "gemini-2.5-pro-preview-05-06"
      }
    },
    "evaluator": {
      "type": "Ragas"
    }
  }

    def get_config(self, component=None):
        if component:
            return self.config.get(component, {})
        return self.config
    def update_config(self, component, config):
        self.config[component] = config
