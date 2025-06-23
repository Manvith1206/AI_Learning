from infrastructure.common.query_classifier_llm import QueryClassifier
import infrastructure.common.rag_constants as constants
from infrastructure.common.rag_constants import (ChunkerType, EmbedderType,RetrieverType, LLMServiceType, RerankerType, EvaluatorType,
                                                 )
from infrastructure.evaluators.custom_evaluator import (
    CustomEvaluator,
    FaithfulnessMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    AnswerRelevancyMetric
)

from infrastructure.evaluators.simple_evaluator  import SimpleEvaluator
from infrastructure.evaluators.ragas_evaluator import RagasEvaluator
from infrastructure.evaluators.deep_eval_evaluator import DeepEval
from config import ConfigManager
from typing import Any
from models import ComponentConfigDetails, ComponentConfigDetail

class ComponentManager:
    def __init__(self, 
                config_manager: ConfigManager, 
                geminiApiKey,
                cohereApiKey, 
                voyageApiKey, 
                mistralApiKey, 
                pineconeApiKey, 
                jinaApiKey, 
                claudeApiKey,
                error_callback,
                warning_callback,
                vector_store):
        
        self.warning_callback = warning_callback
        self.error_callback = error_callback
        self.config_manager = config_manager

        # API Keys assignment (if needed for further use within the pipeline)
        self.geminiApiKey = geminiApiKey
        self.cohereApiKey = cohereApiKey
        self.voyageApiKey = voyageApiKey
        self.mistralApiKey = mistralApiKey
        self.pineconeApiKey = pineconeApiKey
        self.jinaApiKey = jinaApiKey
        self.claudeApiKey = claudeApiKey
        self.vector_store = vector_store
        
    # setup components
    def setup_components(self):
        # Build all core components via factory methods
        self.embedder = self._build_embedder()
        self.chunker = self._build_chunker()
        if self.vector_store is None:
            self.vector_store = self._build_vector_store()
        self.retriever = self._build_retriever()
        self.llm_service = self._build_llm_service()
        self.reranker = self._build_reranker()
        self.evaluator = self._build_evaluator()
        self.query_classifier = QueryClassifier(self.llm_service)
        
        chunker_component = ComponentConfigDetail(config_name=constants.CONFIG_CHUNKER, component=self.chunker)
        embedder_component = ComponentConfigDetail(config_name=constants.CONFIG_EMBEDDER, component=self.embedder)
        vector_store_component = ComponentConfigDetail(config_name=constants.CONFIG_VECTOR_STORE, component=self.vector_store)
        retriever_component = ComponentConfigDetail(config_name=constants.CONFIG_RETRIEVER, component=self.retriever)
        reranker_component = ComponentConfigDetail(config_name=constants.CONFIG_RERANKER, component=self.reranker)
        evaluator_component = ComponentConfigDetail(config_name=constants.CONFIG_EVALUATOR, component=self.evaluator)
        llm_service_component = ComponentConfigDetail(config_name=constants.CONFIG_LLM, component=self.llm_service)

        component_config_details = {}
        
        component_config_details[constants.CONFIG_CHUNKER] = self.chunker
        component_config_details[constants.CONFIG_EMBEDDER] = self.embedder
        component_config_details[constants.CONFIG_VECTOR_STORE] = self.vector_store
        component_config_details[constants.CONFIG_RETRIEVER] = self.retriever
        component_config_details[constants.CONFIG_RERANKER] = self.reranker
        component_config_details[constants.CONFIG_EVALUATOR] = self.evaluator
        component_config_details[constants.CONFIG_LLM] = self.llm_service 

        component_config_details = ComponentConfigDetails(component_config_details)
        print("Setup Components")
        return component_config_details

    def get_chunker_cost_and_time(self):
        return self.chunker.get_cost_and_time_taken()
    def get_embedder_cost_and_time(self):
        return self.embedder.get_cost_and_time_taken()
    def get_vector_store_cost_and_time(self):
        return self.vector_store.get_cost_and_time_taken()
    def get_retriever_cost_and_time(self):
        return self.retriever.get_cost_and_time_taken()
    def get_llm_service_cost_and_time(self):
        return self.llm_service.get_cost_and_time_taken()
    def get_reranker_cost_and_time(self):
        return self.reranker.get_cost_and_time_taken()
    def get_evaluator_cost_and_time(self):
        return self.evaluator.get_cost_and_time_taken()
    
    def get_config_from_config_manager_based_on_config(self, config_name: str):
        config = self.config_manager.get_config(config_name)
        return config
    
    def get_config_param_from_config(self, config: dict[str, Any], config_param: str, defaultValue: list):
        config_param = config.get(config_param, defaultValue)
        return config_param
    
    # build invidual components
    def _build_chunker(self):
        from infrastructure.chunkers.recursive_chunker import RecursiveChunker
        from infrastructure.chunkers.sentence_chunker import SentenceChunker
        from infrastructure.chunkers.semantic_chunker import SemanticChunker
        from infrastructure.chunkers.page_chunker import PageChunker
        from infrastructure.chunkers.semantic_chunker_with_langchain import SemanticChunkerWithLangChain

        config = self.get_config_from_config_manager_based_on_config(constants.CONFIG_CHUNKER)
        config_type = self.get_config_param_from_config(config, constants.CONFIG_TYPE_PARAM, "")
        params = self.get_config_param_from_config(config, constants.CONFIG_PARAM, {})

        if config_type == ChunkerType.RECURSIVE.value:
            return RecursiveChunker(**params)
        elif config_type == ChunkerType.SENTENCE.value:
            return SentenceChunker(**params)
        elif config_type == ChunkerType.SEMANTIC.value:
            return SemanticChunker(**params)
        elif config_type == ChunkerType.PAGE.value:
            return PageChunker()
        elif config_type == ChunkerType.SEMANTIC_WITH_LANGCHAIN.value:
            return SemanticChunkerWithLangChain(self.embedder)
        else:
            return RecursiveChunker()

    def _build_embedder(self):
        from infrastructure.embedders.tfidf_embedder import TFIDFEmbedder
        from infrastructure.embedders.gemini_embedder import GeminiEmbedder
        from infrastructure.embedders.mistral_embedder import MistralEmbedder

        config = self.get_config_from_config_manager_based_on_config(constants.CONFIG_EMBEDDER)
        config_type = self.get_config_param_from_config(config, constants.CONFIG_TYPE_PARAM, "")
        params = self.get_config_param_from_config(config, constants.CONFIG_PARAM, {})
        model_name =  self.get_config_param_from_config(params, constants.CONFIG_MODEL, {})

        if config_type == EmbedderType.TFIDF.value:
            return TFIDFEmbedder()
        elif config_type == EmbedderType.GEMINI.value:
            return GeminiEmbedder(api_key=self.geminiApiKey, model_name=model_name)
        elif config_type == EmbedderType.COHERE.value:
            from infrastructure.embedders.cohere_embedder import CohereEmbedder
            return CohereEmbedder(api_key=self.cohereApiKey,
                                  model=model_name)
        elif config_type == EmbedderType.VOYAGE.value:
            from infrastructure.embedders.voyage_embedder import VoyageEmbedder
            return VoyageEmbedder(api_key=self.voyageApiKey,
                                  model=model_name)
        elif config_type == EmbedderType.MISTRAL.value:
            return MistralEmbedder(api_key=self.mistralApiKey,
                                  model=model_name,
                                  )
        else:
            return TFIDFEmbedder()

    def _build_vector_store(self):
        from infrastructure.vector_stores.pinecone_vector_store import PineConeVectorStore
        from infrastructure.vector_stores.FAISS_Vector_Store import FAISS_Vector_Store
        from infrastructure.vector_stores.sklearn_vector_store import SklearnVectorStore

        config = self.get_config_from_config_manager_based_on_config(constants.CONFIG_VECTOR_STORE)
        config_type = self.get_config_param_from_config(config, constants.CONFIG_TYPE_PARAM, "")
        params = self.get_config_param_from_config(config, constants.CONFIG_PARAM, {})
        
        if config_type == constants.VectorStore.SCIKIT_LEARN.value:
            return SklearnVectorStore(**params)
        elif config_type == constants.VectorStore.PINE_CONE.value:
            return PineConeVectorStore(api_key=self.pineconeApiKey, index_name=constants.PINE_CONE_INDEX_NAME)
        elif config_type == constants.VectorStore.CHROMA.value:
            from infrastructure.vector_stores.chroma_vector_store import ChromaVectorStore
            return ChromaVectorStore(**params, collectionName=constants.CHROMA_COLLECTION_NAME)
        elif config_type == constants.VectorStore.FAISS.value:
            return FAISS_Vector_Store()
        else:
            return SklearnVectorStore(metric=constants.CONFIG_METRIC_COSINE)

    def _build_retriever(self):
        from infrastructure.retrieval_methods.similarity_retriever import SimilarityRetriever
        from infrastructure.retrieval_methods.sentence_window_retreiver import SentenceWindowRetriever
        from infrastructure.retrieval_methods.similarity_retriever import SimilarityRetriever

        config = self.get_config_from_config_manager_based_on_config(constants.CONFIG_RETRIEVER)
        config_type = self.get_config_param_from_config(config, constants.CONFIG_TYPE_PARAM, "")
        params = self.get_config_param_from_config(config, constants.CONFIG_PARAM, {})

        self.top_k = self.get_config_param_from_config(params, constants.CONFIG_TOP_K_PARAM, 0)
        if config_type == RetrieverType.SIMILARITY.value:
            return SimilarityRetriever(**params)
        elif config_type == RetrieverType.HYBRID.value:
            from infrastructure.retrieval_methods.hybrid_retriever import HybridRetriever
            return HybridRetriever(**params)
        elif config_type == RetrieverType.SENTENCE_WINDOW.value:
            return SentenceWindowRetriever(**params)
        else:
            return SimilarityRetriever()

    def _build_llm_service(self):
        
        from infrastructure.llm_chat_services.gemini_service import GeminiService
        from google import genai

        config = self.get_config_from_config_manager_based_on_config(constants.CONFIG_LLM)
        config_type = self.get_config_param_from_config(config, constants.CONFIG_TYPE_PARAM, "")
        params = self.get_config_param_from_config(config, constants.CONFIG_PARAM, {})

        model_name = self.get_config_param_from_config(params, constants.CONFIG_MODEL, "")

        if config_type == LLMServiceType.GEMINI.value:
            client = genai.Client(api_key=self.geminiApiKey)
            return GeminiService(client, model_name=model_name)
        elif config_type == LLMServiceType.CLAUDE.value:
            from infrastructure.llm_chat_services.claude_service import ClaudeService
            import anthropic
            client = anthropic.Anthropic(api_key=self.claudeApiKey)
            return ClaudeService(client, model_name=model_name)
        else:
            return GeminiService(client, model_name=model_name)

    def _build_reranker(self):
        
        config = self.get_config_from_config_manager_based_on_config(constants.CONFIG_RERANKER)
        config_type = self.get_config_param_from_config(config, constants.CONFIG_TYPE_PARAM, "")
        params = self.get_config_param_from_config(config, constants.CONFIG_PARAM, {})

        top_k = self.get_config_param_from_config(params, constants.CONFIG_TOP_K_FOR_RERANKING_PARAM, 0)
        if config_type == RerankerType.LLM.value:
            from infrastructure.rerankers.llm_reranker import LLMReranker

            return LLMReranker(self.llm_service,**params)
        elif config_type == RerankerType.COHERE.value:
            from infrastructure.rerankers.cohere_re_ranker import CohereReranker

            return CohereReranker(self.cohereApiKey, **params)
        elif config_type == RerankerType.JINA.value:
            from infrastructure.rerankers.jina_reranker import JinaReranker

            return JinaReranker(**params, api_key=self.jinaApiKey)
        elif config_type == RerankerType.COSINE.value:
            from infrastructure.rerankers.cosine_reranker import CosineReranker

            return CosineReranker(self.embedder, top_k)
        else:
            from infrastructure.rerankers.cosine_reranker import CosineReranker

            return CosineReranker(self.embedder, top_k_for_reranking=top_k)

    def _build_evaluator(self):
        from infrastructure.evaluators.LLM_Evaluation_Service import LLM_Evaluation_Service
        config = self.get_config_from_config_manager_based_on_config(constants.CONFIG_EVALUATOR)
        config_type = self.get_config_param_from_config(config, constants.CONFIG_TYPE_PARAM, "")
        params = self.get_config_param_from_config(config, constants.CONFIG_PARAM, {})


        if config_type == EvaluatorType.RAGAS.value:
            return RagasEvaluator(**params)
        elif config_type == EvaluatorType.CUSTOM.value:
            try:
                gemini_api_key = self.geminiApiKey
                if not gemini_api_key:
                    self.error_callback(f"Gemini API key ({constants.GEMINI_API_KEY}) not found in st.secrets for Custom Evaluator.")
                    self.warning_callback("Falling back to SimpleEvaluator.")
                    return SimpleEvaluator()
                
                llm_service_for_custom_eval = LLM_Evaluation_Service(client=self.llm_service, 
                                                                     model_name=self.llm_service.model_name,
                                                                     embedder=self.embedder)
                
                metrics_for_custom_eval = [
                    FaithfulnessMetric(llm_service=llm_service_for_custom_eval),
                    ContextPrecisionMetric(llm_service=llm_service_for_custom_eval),
                    ContextRecallMetric(llm_service=llm_service_for_custom_eval),
                    AnswerRelevancyMetric(llm_service=llm_service_for_custom_eval)
                ]
                return CustomEvaluator(metrics=metrics_for_custom_eval)
            except Exception as e:
                self.error_callback(f"Failed to initialize Custom Evaluator: {e}")
                self.error_callback("Falling back to SimpleEvaluator due to an error in Custom Evaluator setup.")
                return SimpleEvaluator()
        elif config_type == EvaluatorType.SIMPLE.value:
            return SimpleEvaluator()
        elif config_type == EvaluatorType.DEEP_EVAL.value:
            return DeepEval(**params, api_key=self.geminiApiKey)
        else:
            return SimpleEvaluator()

    # update components
    def update_component(self, component_name, config):
        self.config_manager.update_config(component_name, config)
        if component_name in [
            constants.CONFIG_CHUNKER,
            constants.CONFIG_EMBEDDER,
            constants.CONFIG_VECTOR_STORE,
            constants.CONFIG_LLM,
            constants.CONFIG_RERANKER
        ]:
            # heavy components: rebuild whole pipeline
            self.setup_components()
        elif component_name == constants.CONFIG_RETRIEVER:
            # hot-swap retriever only
            self.retriever = self._build_retriever()
        elif component_name == constants.CONFIG_EVALUATOR:
            # hot-swap evaluator only
            self.evaluator = self._build_evaluator()
        # else: unknown component, ignore
