import os
import logging
from models import AppConfig
from infrastructure.common.exceptions import ComponentBuildError, MissingConfigurationError
import infrastructure.common.rag_constants as constants

# Import all components to ensure they are registered
import infrastructure.common.component_imports

from infrastructure.common.component_registry import (
    CHUNKERS_REGISTRY,
    EMBEDDERS_REGISTRY,
    VECTOR_STORES_REGISTRY,
    RETRIEVERS_REGISTRY,
    LLM_SERVICES_REGISTRY,
    RERANKERS_REGISTRY,
    EVALUATORS_REGISTRY
)

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentManager:
    """A stateless factory for creating pipeline components based on configuration."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.components = {}

    def build_all_components(self):
        """Builds and returns a dictionary of all pipeline components."""
        logger.info("Building all pipeline components...")
        # Build components in order of dependency
        self.components[constants.CONFIG_EMBEDDER] = self._build_embedder()
        self.components[constants.CONFIG_CHUNKER] = self._build_chunker()
        self.components[constants.CONFIG_VECTOR_STORE] = self._build_vector_store()
        self.components[constants.CONFIG_RETRIEVER] = self._build_retriever()
        self.components[constants.CONFIG_LLM] = self._build_llm_service()
        self.components[constants.CONFIG_RERANKER] = self._build_reranker()
        self.components[constants.CONFIG_EVALUATOR] = self._build_evaluator()
        logger.info("All components built successfully.")
        return self.components

    def _get_api_key(self, key_name: str) -> str:
        """Retrieves an API key from environment variables."""
        api_key = os.getenv(key_name)
        if not api_key:
            raise MissingConfigurationError(f"API key '{key_name}' not found in environment variables.")
        return api_key

    def _build_component(self, registry, config, component_type_name, **kwargs):
        component_type = config.type
        logger.info(f"Building {component_type_name} of type: {component_type}")
        
        component_class = registry.get(component_type)
        if not component_class:
            raise ComponentBuildError(f"Unknown {component_type_name} type: {component_type}")

        try:
            params = {**config.params, **kwargs}
            return component_class(**params)
        except Exception as e:
            raise ComponentBuildError(f"Failed to build {component_type_name}: {e}")

    def _build_chunker(self):
        kwargs = {}
        if self.config.chunker.type == constants.ChunkerType.SEMANTIC.value:
            kwargs['embedder'] = self.components[constants.CONFIG_EMBEDDER]
        return self._build_component(CHUNKERS_REGISTRY, self.config.chunker, "chunker", **kwargs)

    def _build_embedder(self):
        kwargs = {}
        embedder_type = self.config.embedder.type
        if embedder_type == constants.EmbedderType.OPENAI.value:
            kwargs['api_key'] = self._get_api_key(constants.OPENAI_API_KEY)
        elif embedder_type == constants.EmbedderType.COHERE.value:
            kwargs['api_key'] = self._get_api_key(constants.COHERE_API_KEY)
        elif embedder_type == constants.EmbedderType.GEMINI.value:
            kwargs['api_key'] = self._get_api_key(constants.GEMINI_API_KEY)
        elif embedder_type == constants.EmbedderType.MISTRAL.value:
            kwargs['api_key'] = self._get_api_key(constants.MISTRAL_API_KEY)
        elif embedder_type == constants.EmbedderType.VOYAGE.value:
            kwargs['api_key'] = self._get_api_key(constants.VOYAGE_API_KEY)
        return self._build_component(EMBEDDERS_REGISTRY, self.config.embedder, "embedder", **kwargs)

    def _build_vector_store(self):
        kwargs = {}
        if self.config.vector_store.type == constants.VectorStore.PINE_CONE.value:
            kwargs['api_key'] = self._get_api_key(constants.PINECONE_API_KEY)
        return self._build_component(VECTOR_STORES_REGISTRY, self.config.vector_store, "vector store", **kwargs)

    def _build_retriever(self):
        # The vector_store is passed to the retrieve method at runtime, not during initialization.
        kwargs = {}
        return self._build_component(RETRIEVERS_REGISTRY, self.config.retriever, "retriever", **kwargs)

    def _build_llm_service(self):
        kwargs = {}
        llm_type = self.config.llm.type
        if llm_type == constants.LLMServiceType.GEMINI.value:
            kwargs['api_key'] = self._get_api_key(constants.GEMINI_API_KEY)
        elif llm_type == constants.LLMServiceType.OPENAI.value:
            kwargs['api_key'] = self._get_api_key(constants.OPENAI_API_KEY)
        elif llm_type == constants.LLMServiceType.CLAUDE.value:
            kwargs['api_key'] = self._get_api_key(constants.CLAUDE_API_KEY)
        elif llm_type == constants.LLMServiceType.COHERE.value:
            kwargs['api_key'] = self._get_api_key(constants.COHERE_API_KEY)
        return self._build_component(LLM_SERVICES_REGISTRY, self.config.llm, "LLM service", **kwargs)

    def _build_reranker(self):
        kwargs = {}
        reranker_type = self.config.reranker.type
        if reranker_type == constants.RerankerType.LLM.value:
            kwargs['llm_service'] = self.components[constants.CONFIG_LLM]
        elif reranker_type == constants.RerankerType.COHERE.value:
            kwargs['api_key'] = self._get_api_key(constants.COHERE_API_KEY)
        elif reranker_type == constants.RerankerType.JINA.value:
            kwargs['api_key'] = self._get_api_key(constants.JINA_RERANKER_API_KEY)
        return self._build_component(RERANKERS_REGISTRY, self.config.reranker, "reranker", **kwargs)

    def _build_evaluator(self):
        kwargs = {}
        evaluator_type = self.config.evaluator.type
        logger.info(f"Attempting to build evaluator of type: {evaluator_type}")

        evaluator_class = EVALUATORS_REGISTRY.get(evaluator_type)

        if not evaluator_class:
            logger.warning(f"Unknown or unsupported evaluator type '{evaluator_type}'. Defaulting to SimpleEvaluator.")
            evaluator_type = constants.EvaluatorType.SIMPLE.value
            evaluator_class = EVALUATORS_REGISTRY.get(evaluator_type)

        try:
            if evaluator_type == constants.EvaluatorType.CUSTOM.value:
                kwargs['gemini_api_key'] = self._get_api_key(constants.GEMINI_API_KEY)
            elif evaluator_type == constants.EvaluatorType.DEEP_EVAL.value:
                kwargs['gemini_api_key'] = self._get_api_key(constants.GEMINI_API_KEY)
            elif evaluator_type == constants.EvaluatorType.RAGAS.value:
                try:
                    kwargs['openai_api_key'] = self._get_api_key(constants.OPENAI_API_KEY)
                except MissingConfigurationError:
                    kwargs['openai_api_key'] = None
                try:
                    kwargs['gemini_api_key'] = self._get_api_key(constants.GEMINI_API_KEY)
                except MissingConfigurationError:
                    kwargs['gemini_api_key'] = None
                
                if not kwargs.get('openai_api_key') and not kwargs.get('gemini_api_key'):
                    raise MissingConfigurationError("RagasEvaluator requires either an OpenAI or Gemini API key.")

            params = {**self.config.evaluator.params, **kwargs}
            logger.info(f"Building evaluator '{evaluator_type}' with params: {list(params.keys())}")
            return evaluator_class(**params)
            
        except MissingConfigurationError as e:
            logger.warning(f"API key missing for '{evaluator_type}' evaluator: {e}. Falling back to SimpleEvaluator.")
            evaluator_class = EVALUATORS_REGISTRY.get(constants.EvaluatorType.SIMPLE.value)
            return evaluator_class()
        except Exception as e:
            logger.error(f"Failed to build evaluator '{evaluator_type}': {e}. Falling back to SimpleEvaluator.")
            evaluator_class = EVALUATORS_REGISTRY.get(constants.EvaluatorType.SIMPLE.value)
            return evaluator_class()
