import os
from typing import Dict, List
# import uuid # Removed unused import
import logging
import json
import traceback
import re
import uuid

# Setup logger for this module
logger = logging.getLogger(__name__)

from infrastructure.Common.exceptions import (
    PipelineError, ComponentBuildError, MissingConfigurationError, 
    InvalidConfigurationError, EvaluationError, FlashcardGenerationError,
    DocumentProcessingError
)
from infrastructure.Evaluators.simple_evaluator import SimpleEvaluator
from infrastructure.Evaluators.ragas_evaluator import RagasEvaluator
from infrastructure.Evaluators.custom_evaluator import (
    CustomEvaluator,
    FaithfulnessMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    AnswerRelevancyMetric
)
from infrastructure.Evaluators.LLM_Evaluation_Service import LLM_Evaluation_Service # Added import
from config import ConfigManager

from infrastructure.Common.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType, GeminiLLMModel
)
import infrastructure.Common.RAG_Constants as constants
from infrastructure.Common.query_classifier_llm import QueryClassifier

from infrastructure.Evaluators.deep_eval_evaluator import DeepEval 
import logging

class RAGPipeline:
    def __init__(self, config_manager=None, 
                 gemini_api_key=None, anthropic_api_key=None, cohere_api_key=None,
                 openai_api_key=None, pinecone_api_key=None, voyage_api_key=None, mistral_api_key=None, jina_api_key=None):
        self.config_manager = config_manager or ConfigManager()
        
        # Store API keys
        self.gemini_api_key = gemini_api_key
        self.anthropic_api_key = anthropic_api_key
        self.cohere_api_key = cohere_api_key
        self.openai_api_key = openai_api_key # For potential OpenAI services
        self.pinecone_api_key = pinecone_api_key
        self.voyage_api_key = voyage_api_key
        self.mistral_api_key = mistral_api_key
        self.jina_api_key = jina_api_key
        
        self.query_classifier = None
        self.setup_components()
        logging.log(level=0, msg=f"QueryClassigier: {self.query_classifier}, {traceback.print_stack()}")

    # setup components
    def setup_components(self):
        # Build all core components via factory methods
        self.chunker = self.build_chunker()
        self.embedder = self.build_embedder()
        self.vector_store = self.build_vector_store()
        self.retriever = self.build_retriever()
        print("Retriever", self.retriever)
        self.llm_service = self.build_llm_service()
        print("LLMService: ", self.llm_service)
        self.reranker = self.build_reranker()
        self.evaluator = self.build_evaluator()
        self.query_classifier = QueryClassifier(self.llm_service)
        print("QueryClassifier", self.query_classifier)

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
    
    # build invidual components
    def build_chunker(self):
        from infrastructure.Chunkers.recursive_chunker import RecursiveChunker
        from infrastructure.Chunkers.sentence_chunker import SentenceChunker
        from infrastructure.Chunkers.semantic_chunker import SemanticChunker
        from infrastructure.Chunkers.page_chunker import PageChunker
        from infrastructure.Chunkers.semantic_chunker_with_langchain import SemanticChunkerWithLangChain

        cfg = self.config_manager.get_config(constants.CONFIG_CHUNKER)
        type = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {})
        if type == ChunkerType.RECURSIVE.value:
            return RecursiveChunker(**params)
        elif type == ChunkerType.SENTENCE.value:
            return SentenceChunker(**params)
        elif type == ChunkerType.SEMANTIC.value:
            return SemanticChunker(**params)
        elif type == ChunkerType.PAGE.value:
            return PageChunker()
        elif type == ChunkerType.SEMANTIC_WITH_LANGCHAIN.value:
            return SemanticChunkerWithLangChain()
        else:
            return RecursiveChunker()

    def build_embedder(self):
        from infrastructure.Embedders.tfidf_embedder import TFIDFEmbedder
        from infrastructure.Embedders.gemini_embedder import GeminiEmbedder
        from infrastructure.Embedders.mistral_embedder import MistralEmbedder

        cfg = self.config_manager.get_config(constants.CONFIG_EMBEDDER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {})
        model_name = params.get(constants.CONFIG_MODEL)
        if t == EmbedderType.TFIDF.value:
            logger.info("Building TFIDFEmbedder.")
            return TFIDFEmbedder()
        
        elif t == EmbedderType.GEMINI.value:
            logger.info(f"Building GeminiEmbedder for model: {model_name or 'default'}.")
            api_key = self.gemini_api_key
            if not api_key:
                err_msg = f"Gemini API key ('{constants.GEMINI_API_KEY}') not found for GeminiEmbedder. Please provide it during RAGPipeline initialization."
                logger.error(err_msg)
                raise MissingConfigurationError(err_msg)
            try:
                return GeminiEmbedder(api_key=api_key, model_name=model_name)
            except Exception as e:
                logger.error(f"Failed to initialize GeminiEmbedder: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build GeminiEmbedder: {e}")

        elif t == EmbedderType.COHERE.value:
            from infrastructure.Embedders.cohere_embedder import CohereEmbedder
            logger.info(f"Building CohereEmbedder for model: {model_name or 'default'}.")
            api_key = self.cohere_api_key
            if not api_key:
                err_msg = f"Cohere API key ('{constants.COHERE_API_KEY}') not found for CohereEmbedder. Please provide it during RAGPipeline initialization."
                logger.error(err_msg)
                raise MissingConfigurationError(err_msg)
            try:
                return CohereEmbedder(api_key=api_key, model=model_name)
            except Exception as e:
                logger.error(f"Failed to initialize CohereEmbedder: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build CohereEmbedder: {e}")

        elif t == EmbedderType.VOYAGE.value:
            from infrastructure.Embedders.voyage_embedder import VoyageEmbedder
            logger.info(f"Building VoyageEmbedder for model: {model_name or 'default'}.")
            api_key = self.voyage_api_key
            if not api_key:
                err_msg = f"Voyage API key ('{constants.VOYAGE_API_KEY}') not found for VoyageEmbedder. Please provide it during RAGPipeline initialization."
                logger.error(err_msg)
                raise MissingConfigurationError(err_msg)
            try:
                return VoyageEmbedder(api_key=api_key, model=model_name)
            except Exception as e:
                logger.error(f"Failed to initialize VoyageEmbedder: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build VoyageEmbedder: {e}")

        elif t == EmbedderType.MISTRAL.value:
            logger.info(f"Building MistralEmbedder for model: {model_name or 'default'}.")
            api_key = self.mistral_api_key
            if not api_key:
                err_msg = f"Mistral API key ('{constants.MISTRAL_API_KEY}') not found for MistralEmbedder. Please provide it during RAGPipeline initialization."
                logger.error(err_msg)
                raise MissingConfigurationError(err_msg)
            try:
                return MistralEmbedder(api_key=api_key, model=model_name)
            except Exception as e:
                logger.error(f"Failed to initialize MistralEmbedder: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build MistralEmbedder: {e}")

        else:
            logger.warning(f"Unsupported or misconfigured Embedder type: '{t}'. Defaulting to TFIDFEmbedder.")
            return TFIDFEmbedder()

    def build_vector_store(self):
        from infrastructure.Vector_Stores.pinecone_vector_store import PineConeVectorStore
        from infrastructure.Vector_Stores.FAISS_Vector_Store import FAISS_Vector_Store
        from infrastructure.Vector_Stores.sklearn_vector_store import SklearnVectorStore

        cfg = self.config_manager.get_config(constants.CONFIG_VECTOR_STORE)
        params = cfg.get(constants.CONFIG_PARAM, {})
        store_type = cfg.get(constants.CONFIG_TYPE_PARAM)

        if store_type == constants.VectorStore.SCIKIT_LEARN.value:
            logger.info(f"Building SklearnVectorStore with params: {params}")
            try:
                return SklearnVectorStore(**params)
            except Exception as e:
                logger.error(f"Failed to initialize SklearnVectorStore: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build SklearnVectorStore: {e}")

        elif store_type == constants.VectorStore.PINE_CONE.value:
            logger.info(f"Building PineConeVectorStore for index: {constants.PINE_CONE_INDEX_NAME}")
            pinecone_api_key = self.pinecone_api_key
            if not pinecone_api_key:
                err_msg = f"Pinecone API key ('{constants.PINECONE_API_KEY}') not found. Please provide it during RAGPipeline initialization."
                logger.error(err_msg)
                raise MissingConfigurationError(err_msg)
            try:
                return PineConeVectorStore(api_key=pinecone_api_key, index_name=constants.PINE_CONE_INDEX_NAME, **params)
            except Exception as e:
                logger.error(f"Failed to initialize PineConeVectorStore: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build PineConeVectorStore: {e}")

        elif store_type == constants.VectorStore.CHROMA.value:
            from infrastructure.Vector_Stores.chroma_vector_store import ChromaVectorStore
            logger.info(f"Building ChromaVectorStore for collection: {constants.CHROMA_COLLECTION_NAME} with params: {params}")
            try:
                return ChromaVectorStore(**params, collectionName=constants.CHROMA_COLLECTION_NAME)
            except Exception as e:
                logger.error(f"Failed to initialize ChromaVectorStore: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build ChromaVectorStore: {e}")

        elif store_type == constants.VectorStore.FAISS.value:
            logger.info("Building FAISS_Vector_Store.")
            try:
                return FAISS_Vector_Store(**params) # Assuming FAISS might take params like index_path
            except Exception as e:
                logger.error(f"Failed to initialize FAISS_Vector_Store: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build FAISS_Vector_Store: {e}")

        else:
            logger.warning(f"Unsupported or misconfigured VectorStore type: '{store_type}'. Defaulting to SklearnVectorStore with cosine metric.")
            return SklearnVectorStore(metric=constants.CONFIG_METRIC_COSINE)

    def build_retriever(self):
        from infrastructure.Retrieval_Methods.similarity_retriever import SimilarityRetriever
        from infrastructure.Retrieval_Methods.sentence_window_retreiver import SentenceWindowRetriever
        from infrastructure.Retrieval_Methods.similarity_retriever import SimilarityRetriever

        cfg = self.config_manager.get_config(constants.CONFIG_RETRIEVER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {})
        self.top_k = params.get(constants.CONFIG_TOP_K_PARAM, getattr(self, 'top_k', 5))
        if t == RetrieverType.SIMILARITY.value:
            return SimilarityRetriever(**params)
        elif t == RetrieverType.HYBRID.value:
            from infrastructure.Retrieval_Methods.hybrid_retriever import HybridRetriever

            return HybridRetriever(**params)
        # elif t == RetrieverType.SENTENCE_WINDOW.value:
        #     return SentenceWindowRetriever(**params)
        else:
            return SimilarityRetriever()

    def build_llm_service(self):
        from infrastructure.LLM_Chat_Services.gemini_service import GeminiService
        # ClaudeService and CohereChat are imported conditionally below

        cfg = self.config_manager.get_config(constants.CONFIG_LLM)
        llm_service_type = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {}) # Ensure params is a dict to avoid error on .get()
        model_name = params.get(constants.CONFIG_MODEL)

        if llm_service_type == LLMServiceType.GEMINI.value:
            from google import genai # Import specific to Gemini
            gemini_api_key = self.gemini_api_key
            if not gemini_api_key:
                err_msg = f"Gemini API key ('{constants.GEMINI_API_KEY}') not found. Please set it in your environment variables."
                logger.error(err_msg)
                raise MissingConfigurationError(err_msg)
            try:
                client = genai.Client(api_key=gemini_api_key)
                logger.info(f"Successfully initialized Gemini client for model: {model_name or 'default'}")
                return GeminiService(client, model_name=model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build GeminiService: {e}")

        # elif llm_service_type == LLMServiceType.COHERE.value:
        #     from infrastructure.LLM_Chat_Services.cohere_service import CohereChat
        #     cohere_api_key = self.config_manager.get_secret(constants.COHERE_API_KEY)
        #     if not cohere_api_key:
        #         err_msg = f"Cohere API key ('{constants.COHERE_API_KEY}') not found. Please set it in your environment variables."
        #         logger.error(err_msg)
        #         raise MissingConfigurationError(err_msg)
        #     try:
        #         # Assuming CohereChat takes api_key directly, adjust if it needs a client object
        #         logger.info(f"Successfully initialized CohereChat for model: {model_name or 'default'}")
        #         return CohereChat(api_key=cohere_api_key, model_name=model_name)
        #     except Exception as e:
        #         logger.error(f"Failed to initialize CohereChat: {e}", exc_info=True)
        #         raise ComponentBuildError(f"Failed to build CohereChat: {e}")

        elif llm_service_type == LLMServiceType.CLAUDE.value:
            from infrastructure.LLM_Chat_Services.claude_service import ClaudeService
            import anthropic # Import specific to Claude
            anthropic_api_key = self.anthropic_api_key
            if not anthropic_api_key:
                err_msg = f"Anthropic API key ('{constants.ANTHROPIC_API_KEY}') not found for Claude. Please set it in your environment variables."
                logger.error(err_msg)
                raise MissingConfigurationError(err_msg)
            try:
                client = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info(f"Successfully initialized Anthropic client for Claude model: {model_name or 'default'}")
                return ClaudeService(client, model_name=model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client for Claude: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build ClaudeService: {e}")
        
        else:
            err_msg = f"Unsupported or misconfigured LLM service type: '{llm_service_type}'. Check configuration for '{constants.CONFIG_LLM}'."
            logger.error(err_msg)
            raise InvalidConfigurationError(err_msg)

    def build_reranker(self):
        cfg = self.config_manager.get_config(constants.CONFIG_RERANKER)
        reranker_type = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {}) # Ensure params is a dict

        if reranker_type == RerankerType.LLM.value:
            from infrastructure.Rerankers.llm_reranker import LLMReranker
            logger.info(f"Building LLMReranker with params: {params}")
            # LLMReranker uses self.llm_service, which should already be configured
            try:
                return LLMReranker(self.llm_service, **params)
            except Exception as e:
                logger.error(f"Failed to initialize LLMReranker: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build LLMReranker: {e}")
        
        elif reranker_type == RerankerType.COHERE.value:
            from infrastructure.Rerankers.cohere_re_ranker import CohereReranker
            cohere_api_key = self.cohere_api_key # Use instance attribute
            if not cohere_api_key:
                err_msg = f"Cohere API key ('{constants.COHERE_API_KEY}') not found. Please provide it during RAGPipeline initialization for CohereReranker."
                logger.error(err_msg)
                raise MissingConfigurationError(err_msg)
            try:
                logger.info(f"Building CohereReranker with params: {params}")
                return CohereReranker(api_key=cohere_api_key, **params)
            except Exception as e:
                logger.error(f"Failed to initialize CohereReranker: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build CohereReranker: {e}")

        elif reranker_type == RerankerType.JINA.value:
            from infrastructure.Rerankers.jina_reranker import JinaReranker
            logger.info(f"Building JinaReranker with params: {params}")
            try:
                jina_api_key = self.jina_api_key
                if not jina_api_key:
                    err_msg = f"Jina API key ('{constants.JINA_API_KEY}') not found. Please provide it during RAGPipeline initialization for JinaReranker."
                    logger.error(err_msg)
                    raise MissingConfigurationError(err_msg)
                return JinaReranker(api_key=jina_api_key, **params)
            except Exception as e:
                logger.error(f"Failed to initialize JinaReranker: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build JinaReranker: {e}")

        elif reranker_type == RerankerType.COSINE.value:
            from infrastructure.Rerankers.cosine_reranker import CosineReranker
            top_k_cosine = params.get(constants.CONFIG_TOP_K_FOR_RERANKING_PARAM, 5) # Default top_k for cosine if not specified
            logger.info(f"Building CosineReranker with top_k: {top_k_cosine}")
            try:
                return CosineReranker(self.embedder, top_k_for_reranking=top_k_cosine)
            except Exception as e:
                logger.error(f"Failed to initialize CosineReranker: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build CosineReranker: {e}")
            
        else:
            err_msg = f"Unsupported or misconfigured Reranker type: '{reranker_type}'. Check configuration for '{constants.CONFIG_RERANKER}'."
            logger.error(err_msg)
            if reranker_type:
                 raise InvalidConfigurationError(err_msg)
            else:
                logger.warning(f"{err_msg} Defaulting to CosineReranker.")
                from infrastructure.Rerankers.cosine_reranker import CosineReranker
                top_k_default = params.get(constants.CONFIG_TOP_K_FOR_RERANKING_PARAM, 5)
                try:
                    return CosineReranker(self.embedder, top_k_for_reranking=top_k_default)
                except Exception as e:
                    logger.error(f"Failed to initialize default CosineReranker: {e}", exc_info=True)
                    raise ComponentBuildError(f"Failed to build default CosineReranker: {e}")

    def build_evaluator(self):
        # Assuming necessary evaluator classes are imported at the top of the file or available in the scope.
        # e.g.:
        # from infrastructure.Evaluators.LLM_Evaluation_Service import LLM_Evaluation_Service
        # from infrastructure.Evaluators.Ragas_Evaluator import RagasEvaluator
        # from infrastructure.Evaluators.Custom_Evaluator import CustomEvaluator, FaithfulnessMetric, ContextPrecisionMetric, ContextRecallMetric, AnswerRelevancyMetric
        # from infrastructure.Evaluators.Simple_Evaluator import SimpleEvaluator
        # from infrastructure.Evaluators.DeepEval_Evaluator import DeepEvalEvaluator # Example class name

        cfg = self.config_manager.get_config(constants.CONFIG_EVALUATOR)
        evaluator_type = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {}) # Ensure params is a dict

        logger.info(f"Attempting to build evaluator of type: {evaluator_type if evaluator_type else 'Default (SimpleEvaluator)'}")

        if evaluator_type == EvaluatorType.RAGAS.value:
            try:
                logger.info(f"Building RagasEvaluator with params: {params}")
                gemini_api_key = self.gemini_api_key
                openai_api_key = self.openai_api_key
                if not gemini_api_key:
                    err_msg = f"Gemini API key ('{constants.GEMINI_API_KEY}') not found. It may be required by RagasEvaluator. Please provide it during RAGPipeline initialization."
                    logger.error(err_msg)
                    raise MissingConfigurationError(err_msg)
                if not openai_api_key:
                    err_msg = f"OpenAI API key ('{constants.OPENAI_API_KEY}') not found. It may be required by RagasEvaluator. Please provide it during RAGPipeline initialization."
                    logger.error(err_msg)
                    raise MissingConfigurationError(err_msg)
                return RagasEvaluator(gemini_api_key=gemini_api_key, openai_api_key=openai_api_key, **params)
            except Exception as e:
                logger.error(f"Failed to initialize RagasEvaluator: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build RagasEvaluator: {e}")

        elif evaluator_type == EvaluatorType.CUSTOM.value:
            try:
                logger.info(f"Building CustomEvaluator. It will use the main RAGPipeline's LLM service: {self.llm_service.__class__.__name__ if self.llm_service else 'None'}")
                if not self.llm_service:
                    err_msg = "LLM service (self.llm_service) not available or not built, which is required for CustomEvaluator."
                    logger.error(err_msg)
                    raise ComponentBuildError(err_msg)
                
                llm_eval_service_config = params.get("llm_evaluation_service_params", {})
                llm_service_for_custom_eval = LLM_Evaluation_Service(
                    client=self.llm_service,
                    model_name=llm_eval_service_config.get("model_name", self.llm_service.model_name if self.llm_service else None),
                    embedder=self.embedder
                )
                
                metrics_config = params.get("metrics", [
                    {"name": "FaithfulnessMetric", "params": {}},
                    {"name": "ContextPrecisionMetric", "params": {}},
                    {"name": "ContextRecallMetric", "params": {}},
                    {"name": "AnswerRelevancyMetric", "params": {}}
                ])

                metrics_for_custom_eval = []
                for metric_conf in metrics_config:
                    metric_name = metric_conf.get("name")
                    metric_params = metric_conf.get("params", {})
                    if metric_name == "FaithfulnessMetric":
                        metrics_for_custom_eval.append(FaithfulnessMetric(llm_service=llm_service_for_custom_eval, **metric_params))
                    elif metric_name == "ContextPrecisionMetric":
                        metrics_for_custom_eval.append(ContextPrecisionMetric(llm_service=llm_service_for_custom_eval, **metric_params))
                    elif metric_name == "ContextRecallMetric":
                        metrics_for_custom_eval.append(ContextRecallMetric(llm_service=llm_service_for_custom_eval, **metric_params))
                    elif metric_name == "AnswerRelevancyMetric":
                        metrics_for_custom_eval.append(AnswerRelevancyMetric(llm_service=llm_service_for_custom_eval, **metric_params))
                    else:
                        logger.warning(f"Unknown metric '{metric_name}' specified for CustomEvaluator. Skipping.")
                
                if not metrics_for_custom_eval:
                    logger.error("No valid metrics configured or built for CustomEvaluator.")
                    raise ComponentBuildError("CustomEvaluator requires at least one valid metric.")

                custom_evaluator_specific_params = params.get("custom_evaluator_params", {})
                evaluator = CustomEvaluator(metrics=metrics_for_custom_eval, **custom_evaluator_specific_params)
                logger.info("Successfully built CustomEvaluator.")
                return evaluator
            except MissingConfigurationError as mce:
                logger.error(f"Missing configuration during CustomEvaluator build: {mce}", exc_info=True)
                raise
            except ComponentBuildError as cbe:
                logger.error(f"Component build error during CustomEvaluator setup: {cbe}", exc_info=True)
                logger.warning("Falling back to SimpleEvaluator due to CustomEvaluator build failure.")
                return SimpleEvaluator()
            except Exception as e:
                logger.error(f"Unexpected error during CustomEvaluator build: {e}", exc_info=True)
                logger.warning("Falling back to SimpleEvaluator due to an unexpected error in CustomEvaluator build.")
                return SimpleEvaluator()

        elif evaluator_type == EvaluatorType.SIMPLE.value:
            try:
                logger.info(f"Building SimpleEvaluator with params: {params}")
                return SimpleEvaluator(**params)
            except Exception as e:
                logger.error(f"Failed to initialize SimpleEvaluator: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build SimpleEvaluator: {e}")

        elif evaluator_type == EvaluatorType.DEEP_EVAL.value:
            try:
                # DeepEvalEvaluator is configured to use a Gemini model by default in its current state.
                gemini_api_key = self.gemini_api_key
                if not gemini_api_key:
                    err_msg = f"Gemini API key ('{constants.GEMINI_API_KEY}') not found. It is required by DeepEvalEvaluator as it defaults to a Gemini model. Please provide it during RAGPipeline initialization."
                    logger.error(err_msg) # Error because the default DeepEval will fail without it
                    raise MissingConfigurationError(err_msg)
                return DeepEval(gemini_api_key=gemini_api_key, **params)
            except Exception as e:
                logger.error(f"Failed to initialize DeepEvalEvaluator: {e}", exc_info=True)
                raise ComponentBuildError(f"Failed to build DeepEvalEvaluator: {e}")
        
        else:
            err_msg = f"Unsupported or misconfigured Evaluator type: '{evaluator_type}'. Check configuration for '{constants.CONFIG_EVALUATOR}'."
            if evaluator_type: # If a type was specified but not matched
                logger.error(err_msg)
                raise InvalidConfigurationError(err_msg)
            else: # If no type was specified (e.g., empty or None in config)
                logger.warning(f"{err_msg} Defaulting to SimpleEvaluator.")
                try:
                    return SimpleEvaluator()
                except Exception as e:
                    logger.error(f"Failed to initialize default SimpleEvaluator: {e}", exc_info=True)
                    raise ComponentBuildError(f"Failed to build default SimpleEvaluator: {e}")
            logger.warning(f"Unknown or unset evaluator type: {evaluator_type}. Defaulting to SimpleEvaluator.")
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
            self.retriever = self.build_retriever()
        elif component_name == constants.CONFIG_EVALUATOR:
            # hot-swap evaluator only
            self.evaluator = self.build_evaluator()
        # else: unknown component, ignore

    def extract_text(self, file, temp_dir=constants.TEMP_DOCS_DIR):
        try:
            from infrastructure.document_loaders.pdf_loader import PDFLoader
            from infrastructure.document_loaders.docx_loader import DOCXLoader
            from infrastructure.document_loaders.txt_loader import TXTLoader
            from infrastructure.document_loaders.csv_loader import CSVLoader
            loaders = {
                constants.PDF_EXTENSION: PDFLoader(),
                constants.DOCX_EXTENSION: DOCXLoader(),
                constants.TXT_EXTENSION: TXTLoader(),
                constants.CSV_EXTENSION: CSVLoader(),
            }
            os.makedirs(temp_dir, exist_ok=True)
            file_ext = os.path.splitext(file.name)[1].lower()
            file_path = os.path.join(temp_dir, file.name)
            
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            if file_ext in loaders:
                text = loaders[file_ext].load_document(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # # Remove headers and footers
            # text = re.sub(r"DCA2104: Basics of Data Communication Manipal University Jaipur$MUJ$", "", text)

            # # Remove page numbers or line numbers
            # text = re.sub(r"Unit \d+:.*", "", text)
            # text = re.sub(r"\d+\s*$", "", text)

            # # Remove extra whitespaces and newlines
            # text = re.sub(r"\s+", " ", text).strip()
            with open("ExtractedTextFromPdf.txt", "w", encoding="utf-8") as file:
                file.write(text)

            if not text:
                return None, None
            else:
                return text
        except Exception as e:
            logging.error(f"Error extracting text: {e}, Traceback: {traceback.print_exc()}")
            return None, None
         
    # process documents
    def process_document(self, file, texts=None):
        try:
            chunks =  self.chunker.split_text(text=texts)

            documents = []
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                documents.append({
                    constants.ID: doc_id,
                    constants.PAGE_CONTENT: chunk,
                    constants.METADATA: {"source": file.name}
                })
            texts = [doc[constants.PAGE_CONTENT] for doc in documents]
            embeddings = self.embedder.fit(texts)
            
            
            documents = self.vector_store.format_documents(documents)
            self.vector_store.add_embeddings(embeddings, documents)
            print("ProcessDocs / Vector Store", self.vector_store)
            print("ProcessDocs / Vector Store Docs", self.vector_store.documents)
            
            print("Documents Processed Succesfully")

            return documents, chunks
        except Exception as e:
            logging.error(f"Error processing document: {e}, Traceback: {traceback.print_exc()}")
            return None, None

    def greet_user(self, query_text):
        if self.query_classifier.is_greeting(query_text):
            return {
                constants.ANSWER: self.query_classifier.get_greeting_response(),
                constants.CONTEXTS: "",
                constants.RERANK_EXPLANATION: ""
            }
        
    def irrelvant(self, query_text):
        context_docs = self.get_context_docs(query_text)
        if self.query_classifier.is_irrelevant(query_text, context_docs):
            return {
                constants.ANSWER: self.query_classifier.get_irrelevant_question_response(),
                constants.CONTEXTS: context_docs,
                constants.RERANK_EXPLANATION: ""
            }
        
    def get_context_docs(self, query_text, top_k=None):
        print("Vector Store: ", self.vector_store)
        if not hasattr(self.vector_store, 'documents') or not self.vector_store.documents:
            print("Vector Store", self.vector_store)
            print("Documents", len(self.vector_store.documents))
            raise ValueError("No documents processed. Please upload and process a document before querying.")
        # Use configured top_k if not specified
        if top_k is None:
            top_k = self.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.transform([query_text])
        if isinstance(query_embedding, list) and query_embedding:
            first = query_embedding[0]
            if hasattr(first, "values"):
                query_embedding = [e.values for e in query_embedding]
            elif hasattr(first, "embedding"):
                query_embedding = [e.embedding for e in query_embedding]
                    
        results = self.retriever.retrieve(
                query_embedding, 
                self.vector_store.documents, 
                vector_store=self.vector_store,
                query_text=query_text
                )
        retrieved_docs = [result[constants.Document][constants.PAGE_CONTENT] for result in results]
            
        # Use retriever to get relevant documents
        if not retrieved_docs:
            raise ValueError(constants.UNABLE_TO_RETRIEVE_MESSAGE)
        
        # Rerank documents
        reranked_docs, explanation = self.reranker.rerank(query_text, retrieved_docs, top_k=top_k)
        
        
        context_docs = None
        if reranked_docs:
            context_docs = "\n\n".join(reranked_docs)
            context_docs_list = reranked_docs
        else:
            context_docs = "\n\n".join(retrieved_docs)
            context_docs_list = retrieved_docs

        return context_docs, explanation, context_docs_list
    
    def query(self, query_text, history_text, top_k=None):
        try:
            print("Self", self)
            print("QueryClassifier", self.query_classifier)
            if self.query_classifier.is_greeting(query_text):
                print("IsGreeting")
                yield  {
                constants.ANSWER: self.query_classifier.get_greeting_response(),
                constants.CONTEXTS: "",
                constants.RERANK_EXPLANATION: ""
            }
            # query_text = self.rewrite_query(query_text)
            # Ensure documents are available
            
            context_docs, explanation, context_docs_list = self.get_context_docs(query_text)
            if self.query_classifier.is_irrelevant(query_text, context_docs):
                yield  {
                constants.ANSWER: self.query_classifier.get_irrelevant_question_response(),
                constants.CONTEXTS: context_docs,
                constants.RERANK_EXPLANATION: ""
            }
            # Join contexts
            context = "\n\n".join(context_docs)
            with open("Contexts.txt", "w", encoding="utf-8") as file:
                file.write(context)

            # Generate answer
            answer_prompt = f"""
            <system>
            You are a highly detailed assistant that must answer questions based only on the provided context. Do not make up facts or include any information not explicitly supported by the context. If the answer is not present, respond with "The context does not provide enough information to answer this question."
            You are a expert in Digital Data Communications for University Students
            You have knowledge of Digital Data Communication Techniques like Synchronous and Asynchronous transmission and different line configurations
            
            Answer the question directly and concisely using only the provided context. 
            Focus on the specific question asked without adding extra information.
            <system/>

            Your answers must be:
            - Detailed and well-explained (minimum 6 sentences)
            - Faithfully based only on the context
            - Avoid any assumptions or hallucinations
            
            <user>
            # CONTEXT
            # Below are contexts:
            Context:
            {context}

            # QUERY
            Below is the query asked by User:
            
            Question: {query_text}
            </user>

            Chat History:
            {history_text}

            Answer:
            """
            full_response = ""
            for delta in self.llm_service.generate_response(answer_prompt):
                full_response += delta
                yield {
                constants.ANSWER: full_response,
                constants.CONTEXTS: context_docs_list,
                constants.RERANK_EXPLANATION: explanation
            }

            # Save the query data for potential evaluation
            self.last_query = {
                constants.QUESTION: query_text,
                constants.ANSWER: full_response,
                constants.CONTEXTS: context_docs_list
            }
            
            return {
                constants.ANSWER: full_response,
                constants.CONTEXTS: context_docs_list,
                constants.RERANK_EXPLANATION: explanation
            }
        except Exception as e:
            logger.error(f"Error during RAG query execution: {e}", exc_info=True)
            # Depending on desired behavior, could raise a specific QueryError or re-raise PipelineError
            # For now, let the UI handle this as a general pipeline failure if it catches it.
            # Or, if main_page.py is expected to catch specific errors from query:
            # raise QueryError(f"Failed during query processing: {e}") from e
            # For now, let's assume main_page.py handles general exceptions from pipeline.query
            # and this method should signal failure by returning None or raising an exception that main_page expects.
            # Given the original return None, we'll keep that for now, but ideally, it should raise.
            # To align with other refactoring, let's raise a generic PipelineError.
            raise PipelineError(f"Error during RAG query execution: {e}") from e
        
    def evaluate(self, question=None, answer=None, contexts=None, ground_truths=None):
        """Evaluate the RAG system using the configured evaluator
        
        Args:
            question: The question to evaluate (uses last query if None)
            answer: The answer to evaluate (uses last query if None)
            contexts: The contexts to evaluate (uses last query if None)
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Use last query data if not provided
            if hasattr(self, constants.LAST_QUERY) and (question is None or answer is None or contexts is None):
                question = question or self.last_query[constants.QUESTION]
                answer = answer or self.last_query[constants.ANSWER]
                contexts = contexts or self.last_query[constants.CONTEXTS]
            
            if not (question and answer and contexts):
                raise ValueError("No query data available for evaluation")
            
            # Run evaluation
            metrics = self.evaluator.evaluate(question, answer, contexts, ground_truths)
            return metrics
        except ValueError as ve:
            logger.error(f"ValueError during evaluation: {ve}", exc_info=True)
            raise EvaluationError(f"Invalid data for evaluation: {ve}") from ve
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            raise EvaluationError(f"An unexpected error occurred during evaluation: {e}") from e

    def generate_flashcards_from_text(self, text_content: str, num_flashcards: int = 5) -> List[Dict[str, str]]:
        """Generates flashcards from the given text content using the LLM service."""
        if not text_content.strip():
            logger.warning("Cannot generate flashcards from empty content. Input text is blank.")
            return []

        prompt = f"""
        <system>
        You are an expert flashcard creator. Your task is to generate {num_flashcards} distinct and high-quality flashcards (question and answer pairs) based on the provided text content.
        Each flashcard should focus on a key concept or piece of information from the text.
        Questions should be clear and concise.
        Answers should be accurate and directly derivable from the text.

        Respond ONLY with a valid JSON array of objects. Each object must have two keys: "question" and "answer".
        Do NOT include any other text, explanations, or apologies before or after the JSON array.
        Example format:
        [        
          {{"question": "What is the main topic of the text?", "answer": "The main topic is..."}},
          {{"question": "Define the term 'XYZ'.", "answer": "XYZ is defined as..."}}
        ]
        </system>

        <user>
        # TEXT CONTENT
        {text_content}

        # TASK
        Generate {num_flashcards} flashcards in the specified JSON format based on the text content above.
        </user>

        JSON Output:
        """
        
        try:
            full_response = ""
            # Assuming llm_service.generate_response is a generator yielding response chunks
            for delta in self.llm_service.generate_response(prompt):
                full_response += delta
            
            # Attempt to parse the LLM's response as JSON
            # The response might be wrapped in markdown code blocks, try to strip them
            if full_response.strip().startswith("```json"):
                full_response = full_response.strip()[7:-3].strip()
            elif full_response.strip().startswith("```"):
                 full_response = full_response.strip()[3:-3].strip()

            flashcards = json.loads(full_response)
            
            # Validate structure
            if not isinstance(flashcards, list):
                raise ValueError("LLM response is not a list.")
            for card in flashcards:
                if not (isinstance(card, dict) and "question" in card and "answer" in card):
                    raise ValueError("Invalid flashcard structure in LLM response.")
            
            return flashcards[:num_flashcards] # Return up to the requested number

        except json.JSONDecodeError as e:
            error_message = f"Error decoding JSON from LLM for flashcards. Raw response: {full_response[:500]}..."
            logger.error(f"{error_message} Details: {e}", exc_info=True)
            raise FlashcardGenerationError(error_message) from e
        except ValueError as e:
            error_message = f"Error in flashcard data structure from LLM. Raw response: {full_response[:500]}..."
            logger.error(f"{error_message} Details: {e}", exc_info=True)
            raise FlashcardGenerationError(error_message) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during flashcard generation: {e}", exc_info=True)
            raise FlashcardGenerationError(f"An unexpected error occurred during flashcard generation: {e}") from e
