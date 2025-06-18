import os
from typing import Dict, List
import uuid

from infrastructure.Evaluators.simple_evaluator  import SimpleEvaluator
from infrastructure.Evaluators.ragas_evaluator import RagasEvaluator
from infrastructure.Evaluators.custom_evaluator import (
    CustomEvaluator,
    FaithfulnessMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    AnswerRelevancyMetric
)
from config import ConfigManager
import re

from infrastructure.Common.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType
)
import infrastructure.Common.RAG_Constants as constants
from infrastructure.LLM_Chat_Services.cohere_service import CohereChat
from infrastructure.Common.query_classifier_llm import QueryClassifier

import traceback
import json # Added for parsing LLM response for flashcards
from infrastructure.Evaluators.deep_eval_evaluator import DeepEval
import Utils.Exceptions as Exceptions
from infrastructure.PromptProviders.LLM_Chat_Prompt_Provider import LLM_Chat_Prompt_Provider
from infrastructure.PromptProviders.flashcards_generation_prompt_provider import FlashCardsGeneration_Prompt_Provider

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
                process_doc_callback, 
                config_manager=None, 
                vector_store=None):
        
        self.config_manager = config_manager or ConfigManager()
        self.vector_store = vector_store
        self.warning_callback = warning_callback
        self.error_callback = error_callback
        self.process_doc_callback = process_doc_callback
        
        # API Keys assignment (if needed for further use within the pipeline)
        self.geminiApiKey = geminiApiKey
        self.cohereApiKey = cohereApiKey
        self.voyageApiKey = voyageApiKey
        self.mistralApiKey = mistralApiKey
        self.pineconeApiKey = pineconeApiKey
        self.jinaApiKey = jinaApiKey
        self.claudeApiKey = claudeApiKey
        
        self.query_classifier = None
        self.flashcard_prompt_provider = FlashCardsGeneration_Prompt_Provider()
        self.setup_components()

    # setup components
    def setup_components(self):
        # Build all core components via factory methods
        self.chunker = self._build_chunker()
        self.embedder = self._build_embedder()
        if self.vector_store is None:
            self.vector_store = self._build_vector_store()
        self.retriever = self._build_retriever()
        self.llm_service = self._build_llm_service()
        self.reranker = self._build_reranker()
        self.evaluator = self._build_evaluator()
        self.query_classifier = QueryClassifier(self.llm_service)
        print("Setup Components")

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
    def _build_chunker(self):
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

    def _build_embedder(self):
        from infrastructure.Embedders.tfidf_embedder import TFIDFEmbedder
        from infrastructure.Embedders.gemini_embedder import GeminiEmbedder
        from infrastructure.Embedders.mistral_embedder import MistralEmbedder

        cfg = self.config_manager.get_config(constants.CONFIG_EMBEDDER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {})
        model_name = params.get(constants.CONFIG_MODEL)
        if t == EmbedderType.TFIDF.value:
            return TFIDFEmbedder()
        elif t == EmbedderType.GEMINI.value:
            return GeminiEmbedder(api_key=self.geminiApiKey, model_name=model_name)
        elif t == EmbedderType.COHERE.value:
            from infrastructure.Embedders.cohere_embedder import CohereEmbedder
            return CohereEmbedder(api_key=self.cohereApiKey,
                                  model=model_name)
        elif t == EmbedderType.VOYAGE.value:
            from infrastructure.Embedders.voyage_embedder import VoyageEmbedder
            return VoyageEmbedder(api_key=self.voyageApiKey,
                                  model=model_name)
        elif t == EmbedderType.MISTRAL.value:
            return MistralEmbedder(api_key=self.mistralApiKey,
                                  model=model_name,
                                  )
        else:
            return TFIDFEmbedder()

    def _build_vector_store(self):
        from infrastructure.Vector_Stores.pinecone_vector_store import PineConeVectorStore
        from infrastructure.Vector_Stores.FAISS_Vector_Store import FAISS_Vector_Store
        from infrastructure.Vector_Stores.sklearn_vector_store import SklearnVectorStore

        cfg = self.config_manager.get_config(constants.CONFIG_VECTOR_STORE)
        params = cfg.get(constants.CONFIG_PARAM, {})
        type = cfg.get(constants.CONFIG_TYPE_PARAM)
        api_key = self.pineconeApiKey
        if type == constants.VectorStore.SCIKIT_LEARN.value:
            return SklearnVectorStore(**params)
        elif type == constants.VectorStore.PINE_CONE.value:
            return PineConeVectorStore(api_key=api_key, index_name=constants.PINE_CONE_INDEX_NAME)
        elif type == constants.VectorStore.CHROMA.value:
            from infrastructure.Vector_Stores.chroma_vector_store import ChromaVectorStore
            return ChromaVectorStore(**params, collectionName=constants.CHROMA_COLLECTION_NAME)
        elif type == constants.VectorStore.FAISS.value:
            return FAISS_Vector_Store()
        else:
            return SklearnVectorStore(metric=constants.CONFIG_METRIC_COSINE)

    def _build_retriever(self):
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
        elif t == RetrieverType.SENTENCE_WINDOW.value:
            return SentenceWindowRetriever(**params)
        else:
            return SimilarityRetriever()

    def _build_llm_service(self):
        
        from infrastructure.LLM_Chat_Services.gemini_service import GeminiService

        cfg = self.config_manager.get_config(constants.CONFIG_LLM)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        from google import genai
        client = genai.Client(api_key=self.geminiApiKey)
        params = cfg.get(constants.CONFIG_PARAM)
        model_name = params.get(constants.CONFIG_MODEL)

        if t == LLMServiceType.GEMINI.value:
            return GeminiService(client, model_name=model_name)
        # elif t == LLMServiceType.COHERE.value:
        #     return CohereChat(st.secrets[constants.COHERE_API_KEY], model_name=model_name)
        elif t == LLMServiceType.CLAUDE.value:
            from infrastructure.LLM_Chat_Services.claude_service import ClaudeService
            import anthropic
            client = anthropic.Anthropic(api_key=self.claudeApiKey)
            return ClaudeService(client, model_name=model_name)
        else:
            return GeminiService(client, model_name=model_name)

    def _build_reranker(self):
        cfg = self.config_manager.get_config(constants.CONFIG_RERANKER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM)
        model = params.get(constants.CONFIG_MODEL)
        top_k = params.get(constants.CONFIG_TOP_K_FOR_RERANKING_PARAM)
        if t == RerankerType.LLM.value:
            from infrastructure.Rerankers.llm_reranker import LLMReranker

            return LLMReranker(self.llm_service,**params)
        elif t == RerankerType.COHERE.value:
            from infrastructure.Rerankers.cohere_re_ranker import CohereReranker

            return CohereReranker(self.cohereApiKey, **params)
        elif t == RerankerType.JINA.value:
            from infrastructure.Rerankers.jina_reranker import JinaReranker

            return JinaReranker(**params)
        elif t == RerankerType.COSINE.value:
            from infrastructure.Rerankers.cosine_reranker import CosineReranker

            return CosineReranker(self.embedder, top_k)
        else:
            from infrastructure.Rerankers.cosine_reranker import CosineReranker

            return CosineReranker(self.embedder, top_k_for_reranking=top_k)

    def _build_evaluator(self):
        from infrastructure.Evaluators.LLM_Evaluation_Service import LLM_Evaluation_Service
        cfg = self.config_manager.get_config(constants.CONFIG_EVALUATOR)
        evaluator_type = cfg.get(constants.CONFIG_TYPE_PARAM)

        if evaluator_type == EvaluatorType.RAGAS.value:
            return RagasEvaluator(**cfg.get(constants.CONFIG_PARAM, {}))
        elif evaluator_type == EvaluatorType.CUSTOM.value:
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
        elif evaluator_type == EvaluatorType.SIMPLE.value:
            return SimpleEvaluator()
        elif evaluator_type == EvaluatorType.DEEP_EVAL.value:
            return DeepEval(**cfg.get(constants.CONFIG_PARAM, {}))
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

    def extractText(self, file, temp_dir=constants.TEMP_DOCS_DIR):
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
            self.error_callback(f"Error extracting text: {e}, Traceback: {traceback.print_exc()}")
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
            self.vector_store.documents = documents  # Attach documents for caching
            self.process_doc_callback(f"Document Processed Succesfully with Chunks: {len(chunks)}")
            
            return self.vector_store
        except Exception as e:
            full_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.error_callback(f"Error processing document: {e}, Traceback: {full_traceback}")
            return None
    
    def greetUser(self, query_text):
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
        if not hasattr(self.vector_store, 'documents') or not self.vector_store.documents:
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
        retrieved_docs = [result for result in results]
            
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
            print("Query")
            if self.query_classifier.is_greeting(query_text):
                # UIComponents.create_subheader_UI(self.query_classifier.get_greeting_response())
                # UIComponents.add_message_to_chat("assistant",  self.query_classifier.get_greeting_response())
                yield {
                constants.ANSWER: self.query_classifier.get_greeting_response(),
                constants.CONTEXTS: "",
                constants.RERANK_EXPLANATION: ""
            }
                return
            # query_text = self.rewrite_query(query_text)
            # Ensure documents are available
            
            context_docs, explanation, context_docs_list = self.get_context_docs(query_text)
            if self.query_classifier.is_irrelevant(query_text, context_docs):
                # UIComponents.create_subheader_UI(self.query_classifier.get_irrelevant_question_response())
                # UIComponents.add_message_to_chat("assistant",  self.query_classifier.get_irrelevant_question_response())
                yield {
                constants.ANSWER: self.query_classifier.get_irrelevant_question_response(),
                constants.CONTEXTS: context_docs,
                constants.RERANK_EXPLANATION: ""
            }
                return
            # Join contexts
            context = "\n\n".join(context_docs)
            with open("Contexts.txt", "w", encoding="utf-8") as file:
                file.write(context)

            llm_chat_prompt_provider = LLM_Chat_Prompt_Provider()
            # Generate answer
            answer_prompt = llm_chat_prompt_provider.get_final_prompt(context=context, query_text=query_text, history_text=history_text)
            print("AnswerPrompt: ", answer_prompt)
            full_response = ""
            for delta in self.llm_service.generate_response(answer_prompt):
                full_response += delta
                yield {
                constants.ANSWER: full_response,
                constants.CONTEXTS: context_docs,
                constants.RERANK_EXPLANATION: ""
            }

            # Save the query data for potential evaluation
            self.last_query = {
                constants.QUESTION: query_text,
                constants.ANSWER: full_response,
                constants.CONTEXTS: context_docs_list
            }
            
        except Exception as e:
            self.error_callback(f"Error during query: {e}, Traceback: {traceback.print_exc()}")
            return None
        
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
        except Exception as e:
            raise Exceptions.EvaluationError("Error During Evaluation")

    def generate_flashcards_from_text(self, text_content: str, num_flashcards: int = 5) -> List[Dict[str, str]]:
        """Generates flashcards from the given text content using the LLM service."""
        if not text_content.strip():
            self.warning_callback("Cannot generate flashcards from empty content.")
            return []

        prompt = self.flashcard_prompt_provider.get_final_prompt(text_content=text_content, num_flashcards=num_flashcards)
        
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
            self.error_callback(f"Error decoding JSON from LLM for flashcards: {e}\nRaw response: {full_response}")
            print(f"JSONDecodeError: {e}. Raw LLM response for flashcards:\n{full_response}")
            return []
        except ValueError as e:
            self.error_callback(f"Error in flashcard data structure from LLM: {e}\nRaw response: {full_response}")
            print(f"ValueError: {e}. Raw LLM response for flashcards:\n{full_response}")
            return []
        except Exception as e:
            self.error_callback(f"An unexpected error occurred during flashcard generation: {e}")
            print(f"Unexpected error in generate_flashcards_from_text: {e}, Traceback: {traceback.format_exc()}")
            return []
