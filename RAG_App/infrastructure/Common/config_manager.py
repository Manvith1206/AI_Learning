import os
from infrastructure.common.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType, GeminiLLMModel
)
import infrastructure.common.RAG_Constants as constants

class ConfigManager:
    """Manages configuration for RAG components"""
    def __init__(self, config=None):
        self.config = config or {
            constants.ConfigManagerNames.CONFIG_CHUNKER: {constants.ConfigManagerNames.CONFIG_TYPE_PARAM: ChunkerType.RECURSIVE.value, constants.ConfigManagerNames.CONFIG_PARAM: {constants.ConfigManagerNames.CONFIG_CHUNK_SIZE_PARAM: 600, constants.ConfigManagerNames.CONFIG_CHUNK_OVERLAP_PARAM: 200}},
            constants.ConfigManagerNames.CONFIG_EMBEDDER: {constants.ConfigManagerNames.CONFIG_TYPE_PARAM: EmbedderType.TFIDF.value, constants.ConfigManagerNames.CONFIG_PARAM: {constants.ConfigManagerNames.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBED_MODEL_DEFAULT.value}},
            constants.ConfigManagerNames.CONFIG_VECTOR_STORE: {constants.ConfigManagerNames.CONFIG_TYPE_PARAM: constants.VectorStore.SCIKIT_LEARN.value, constants.ConfigManagerNames.CONFIG_PARAM: {constants.ConfigManagerNames.CONFIG_VECTOR_STORE_METRIC: constants.ConfigManagerNames.CONFIG_METRIC_COSINE}},
            constants.ConfigManagerNames.CONFIG_RETRIEVER: {constants.ConfigManagerNames.CONFIG_TYPE_PARAM: RetrieverType.SIMILARITY.value, constants.ConfigManagerNames.CONFIG_PARAM: {constants.ConfigManagerNames.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0, constants.ConfigManagerNames.CONFIG_TOP_K_PARAM: 5}},
            constants.ConfigManagerNames.CONFIG_RERANKER: {constants.ConfigManagerNames.CONFIG_TYPE_PARAM: RerankerType.LLM.value, constants.ConfigManagerNames.CONFIG_PARAM: {constants.ConfigManagerNames.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5}},
            constants.ConfigManagerNames.CONFIG_LLM: {constants.ConfigManagerNames.CONFIG_TYPE_PARAM: LLMServiceType.CLAUDE.value, constants.ConfigManagerNames.CONFIG_PARAM: {constants.ConfigManagerNames.CONFIG_MODEL: constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_7.display_name}},
            constants.ConfigManagerNames.CONFIG_EVALUATOR: {constants.ConfigManagerNames.CONFIG_TYPE_PARAM: EvaluatorType.RAGAS.value}
        }

    def get_config(self, component=None):
        if component:
            return self.config.get(component, {})
        return self.config
    def update_config(self, component, config):
        self.config[component] = config
