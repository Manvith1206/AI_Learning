import os
from typing import Dict, List
import uuid

from infrastructure.evaluators.simple_evaluator  import SimpleEvaluator
from infrastructure.evaluators.ragas_evaluator import RagasEvaluator

from config import ConfigManager

import infrastructure.common.rag_constants as constants
from infrastructure.common.query_classifier_llm import QueryClassifier

import traceback
import json # Added for parsing LLM response for flashcards
from infrastructure.evaluators.deep_eval_evaluator import DeepEval
import Utils.exceptions as Exceptions
from infrastructure.prompt_providers.llm_chat_prompt_provider import LLM_Chat_Prompt_Provider
from infrastructure.prompt_providers.flashcards_generation_prompt_provider import FlashCardsGeneration_Prompt_Provider
from infrastructure.common.pipelines.component_manager import ComponentManager
from infrastructure.common.pipelines.document_processing import DocumentProcessing
from infrastructure.common.pipelines.flash_cards_generation import FlashCardGeneration
from infrastructure.common.pipelines.query_evaluation import QueryEvaluation
from infrastructure.common.pipelines.query_processing import QueryProcessing

class RAGPipeline:
    def __init__(self, 
                geminiApiKey,
                cohereApiKey, 
                voyageApiKey, 
                mistralApiKey, 
                pineconeApiKey, 
                jinaApiKey, 
                claudeApiKey, 
                warning_callback, 
                error_callback, 
                vector_store,
                process_doc_callback, 
                config_manager=None): 
        
        self.config_manager = config_manager or ConfigManager()
        self.vector_store = vector_store
        self.warning_callback = warning_callback
        self.error_callback = error_callback
        self.process_doc_callback = process_doc_callback

        self.component_manager = ComponentManager(
            config_manager=self.config_manager,
            geminiApiKey = geminiApiKey,
            cohereApiKey = cohereApiKey,
            voyageApiKey = voyageApiKey,
            mistralApiKey = mistralApiKey,
            pineconeApiKey = pineconeApiKey,
            jinaApiKey = jinaApiKey,
            claudeApiKey = claudeApiKey,
            error_callback=self.error_callback,
            warning_callback=self.warning_callback,
            vector_store=None
        )
        components = self.component_manager.setup_components()
        if self.vector_store:
            print("VSDocs", self.vector_store.documents)
        print("ChunkerConfig: ", components.components[constants.CONFIG_VECTOR_STORE].documents)

        self.document_processing = DocumentProcessing(
            error_callback=self.error_callback,
            process_doc_callback=self.process_doc_callback,
            vector_store=components.components[constants.CONFIG_VECTOR_STORE],
            chunker=components.components[constants.CONFIG_CHUNKER],
            embedder=components.components[constants.CONFIG_EMBEDDER]
        )

        self.flashcards_generation = FlashCardGeneration(
            warning_callback=self.warning_callback,
            error_callback=self.error_callback,
            llm_service=components.components[constants.CONFIG_LLM]
        )

        self.query_evaluation = QueryEvaluation(evaluator=components.components[constants.CONFIG_EVALUATOR])

        self.query_processing = QueryProcessing(
            llm_service=components.components[constants.CONFIG_LLM],
            vector_store=components.components[constants.CONFIG_VECTOR_STORE],
            embedder=components.components[constants.CONFIG_EMBEDDER],
            retriever=components.components[constants.CONFIG_RETRIEVER],
            reranker=components.components[constants.CONFIG_RERANKER],
            error_callback=self.error_callback
        )

    