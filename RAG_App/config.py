from infrastructure.Common.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType, GeminiLLMModel
)

import infrastructure.Common.RAG_Constants as constants

class ConfigManager:
    """Manages configuration for RAG components"""
    def __init__(self, config=None):
        self.config = config or {
            constants.CONFIG_CHUNKER: {constants.CONFIG_TYPE_PARAM: ChunkerType.RECURSIVE.value, constants.CONFIG_PARAM: {constants.CONFIG_CHUNK_SIZE_PARAM: 600, constants.CONFIG_CHUNK_OVERLAP_PARAM: 200}},
            constants.CONFIG_EMBEDDER: {constants.CONFIG_TYPE_PARAM: EmbedderType.TFIDF.value, constants.CONFIG_PARAM: {constants.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBED_MODEL_DEFAULT.value}},
            constants.CONFIG_VECTOR_STORE: {constants.CONFIG_TYPE_PARAM: constants.VectorStore.SCIKIT_LEARN.value, constants.CONFIG_PARAM: {constants.CONFIG_VECTOR_STORE_METRIC: constants.CONFIG_METRIC_COSINE}},
            constants.CONFIG_RETRIEVER: {constants.CONFIG_TYPE_PARAM: RetrieverType.SIMILARITY.value, constants.CONFIG_PARAM: {constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0, constants.CONFIG_TOP_K_PARAM: 5}},
            constants.CONFIG_RERANKER: {constants.CONFIG_TYPE_PARAM: RerankerType.LLM.value, constants.CONFIG_PARAM: {constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5}},
            constants.CONFIG_LLM: {constants.CONFIG_TYPE_PARAM: LLMServiceType.CLAUDE.value, constants.CONFIG_PARAM: {constants.CONFIG_MODEL: constants.CLAUDE_MODELS.CLAUDE_OPUS_THREE.value}},
            constants.CONFIG_EVALUATOR: {constants.CONFIG_TYPE_PARAM: EvaluatorType.RAGAS.value}
        }

    def get_config(self, component=None):
        if component:
            return self.config.get(component, {})
        return self.config
    
    def update_config(self, component, config):
        self.config[component] = config
