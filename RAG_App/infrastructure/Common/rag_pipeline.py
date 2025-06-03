import os
import uuid

from RAG_App.infrastructure.Evaluators.simple_evaluator import SimpleEvaluator
from RAG_App.infrastructure.Evaluators.ragas_evaluator import RagasEvaluator
from RAG_App.infrastructure.Evaluators.custom_evaluator import (
    CustomEvaluator,
    FaithfulnessMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    AnswerRelevancyMetric
)
from RAG_App.config import ConfigManager
import streamlit as st
import re

from RAG_App.infrastructure.Common.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType, GeminiLLMModel
)
import RAG_App.infrastructure.Common.RAG_Constants as constants
from RAG_App.infrastructure.LLM_Chat_Services.cohere_service import CohereChat
from RAG_App.infrastructure.Common.query_classifier_llm import QueryClassifier

import traceback
from RAG_App.infrastructure.Evaluators.deep_eval_evaluator import DeepEval
from RAG_App.infrastructure.Vector_Stores.FAISS_Vector_Store import FAISS_Vector_Store # Added for caching

class RAGPipeline:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or ConfigManager()
        self.setup_components()
        self.query_classifier = None

    # setup components
    def setup_components(self):
        # Build all core components via factory methods
        self.chunker = self._build_chunker()
        self.embedder = self._build_embedder()
        self.vector_store = self._build_vector_store()
        self.retriever = self._build_retriever()
        self.llm_service = self._build_llm_service()
        self.reranker = self._build_reranker()
        self.evaluator = self._build_evaluator()
        self.query_classifier = QueryClassifier(self.llm_service)

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
        from RAG_App.infrastructure.Chunkers.recursive_chunker import RecursiveChunker
        from RAG_App.infrastructure.Chunkers.sentence_chunker import SentenceChunker
        from RAG_App.infrastructure.Chunkers.semantic_chunker import SemanticChunker
        from RAG_App.infrastructure.Chunkers.page_chunker import PageChunker
        from RAG_App.infrastructure.Chunkers.semantic_chunker_with_langchain import SemanticChunkerWithLangChain

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
        from RAG_App.infrastructure.Embedders.tfidf_embedder import TFIDFEmbedder
        from RAG_App.infrastructure.Embedders.gemini_embedder import GeminiEmbedder
        from RAG_App.infrastructure.Embedders.mistral_embedder import MistralEmbedder

        cfg = self.config_manager.get_config(constants.CONFIG_EMBEDDER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {})
        model_name = params.get(constants.CONFIG_MODEL)
        if t == EmbedderType.TFIDF.value:
            return TFIDFEmbedder()
        elif t == EmbedderType.GEMINI.value:
            return GeminiEmbedder(api_key=st.secrets[constants.GEMINI_API_KEY], model_name=model_name)
        elif t == EmbedderType.COHERE.value:
            from RAG_App.infrastructure.Embedders.cohere_embedder import CohereEmbedder
            return CohereEmbedder(api_key=st.secrets[constants.COHERE_API_KEY],
                                  model=model_name)
        elif t == EmbedderType.VOYAGE.value:
            from RAG_App.infrastructure.Embedders.voyage_embedder import VoyageEmbedder
            return VoyageEmbedder(api_key=st.secrets[constants.VOYAGE_API_KEY],
                                  model=model_name)
        elif t == EmbedderType.MISTRAL.value:
            return MistralEmbedder(api_key=st.secrets[constants.MISTRAL_API_KEY],
                                  model=model_name,
                                  )
        else:
            return TFIDFEmbedder()

    def _build_vector_store(self):
        from RAG_App.infrastructure.Vector_Stores.pinecone_vector_store import PineConeVectorStore
        from RAG_App.infrastructure.Vector_Stores.FAISS_Vector_Store import FAISS_Vector_Store
        from RAG_App.infrastructure.Vector_Stores.sklearn_vector_store import SklearnVectorStore

        cfg = self.config_manager.get_config(constants.CONFIG_VECTOR_STORE)
        params = cfg.get(constants.CONFIG_PARAM, {})
        type = cfg.get(constants.CONFIG_TYPE_PARAM)
        api_key = st.secrets[constants.PINECONE_API_KEY]
        if type == constants.VectorStore.SCIKIT_LEARN.value:
            return SklearnVectorStore(**params)
        elif type == constants.VectorStore.PINE_CONE.value:
            return PineConeVectorStore(api_key=api_key, index_name=constants.PINE_CONE_INDEX_NAME)
        elif type == constants.VectorStore.CHROMA.value:
            from RAG_App.infrastructure.Vector_Stores.chroma_vector_store import ChromaVectorStore
            return ChromaVectorStore(**params, collectionName=constants.CHROMA_COLLECTION_NAME)
        elif type == constants.VectorStore.FAISS.value:
            return FAISS_Vector_Store()
        else:
            return SklearnVectorStore(metric=constants.CONFIG_METRIC_COSINE)

    def _build_retriever(self):
        from RAG_App.infrastructure.Retrieval_Methods.similarity_retriever import SimilarityRetriever
        from RAG_App.infrastructure.Retrieval_Methods.sentence_window_retreiver import SentenceWindowRetriever
        from RAG_App.infrastructure.Retrieval_Methods.similarity_retriever import SimilarityRetriever

        cfg = self.config_manager.get_config(constants.CONFIG_RETRIEVER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {})
        self.top_k = params.get(constants.CONFIG_TOP_K_PARAM, getattr(self, 'top_k', 5))
        if t == RetrieverType.SIMILARITY.value:
            return SimilarityRetriever(**params)
        elif t == RetrieverType.HYBRID.value:
            from RAG_App.infrastructure.Retrieval_Methods.hybrid_retriever import HybridRetriever

            return HybridRetriever(**params)
        # elif t == RetrieverType.SENTENCE_WINDOW.value:
        #     return SentenceWindowRetriever(**params)
        else:
            return SimilarityRetriever()

    def _build_llm_service(self):
        
        from RAG_App.infrastructure.LLM_Chat_Services.gemini_service import GeminiService

        cfg = self.config_manager.get_config(constants.CONFIG_LLM)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        from google import genai
        client = genai.Client(api_key=st.secrets[constants.GEMINI_API_KEY])
        params = cfg.get(constants.CONFIG_PARAM)
        model_name = params.get(constants.CONFIG_MODEL)

        if t == LLMServiceType.GEMINI.value:
            return GeminiService(client, model_name=model_name)
        elif t == LLMServiceType.COHERE.value:
            return CohereChat(st.secrets[constants.COHERE_API_KEY], model_name=model_name)
        elif t == LLMServiceType.CLAUDE.value:
            from RAG_App.infrastructure.LLM_Chat_Services.claude_service import ClaudeService
            import anthropic
            client = anthropic.Anthropic(api_key=st.secrets[constants.CLAUDE_API_KEY])
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
            from RAG_App.infrastructure.Rerankers.llm_reranker import LLMReranker

            return LLMReranker(self.llm_service,**params)
        elif t == RerankerType.COHERE.value:
            from RAG_App.infrastructure.Rerankers.cohere_re_ranker import CohereReranker

            return CohereReranker(st.secrets[constants.COHERE_API_KEY], **params)
        elif t == RerankerType.JINA.value:
            from RAG_App.infrastructure.Rerankers.jina_reranker import JinaReranker

            return JinaReranker(**params)
        elif t == RerankerType.COSINE.value:
            from RAG_App.infrastructure.Rerankers.cosine_reranker import CosineReranker

            return CosineReranker(self.embedder, top_k)
        else:
            from RAG_App.infrastructure.Rerankers.cosine_reranker import CosineReranker

            return CosineReranker(self.embedder, top_k_for_reranking=top_k)

    def _build_evaluator(self):
        from RAG_App.infrastructure.Evaluators.LLM_Evaluation_Service import LLM_Evaluation_Service
        cfg = self.config_manager.get_config(constants.CONFIG_EVALUATOR)
        evaluator_type = cfg.get(constants.CONFIG_TYPE_PARAM)

        if evaluator_type == EvaluatorType.RAGAS.value:
            return RagasEvaluator(**cfg.get(constants.CONFIG_PARAM, {}))
        elif evaluator_type == EvaluatorType.CUSTOM.value:
            try:
                gemini_api_key = st.secrets.get(constants.GEMINI_API_KEY)
                if not gemini_api_key:
                    st.error(f"Gemini API key ({constants.GEMINI_API_KEY}) not found in st.secrets for Custom Evaluator.")
                    st.warning("Falling back to SimpleEvaluator.")
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
                st.error(f"Failed to initialize Custom Evaluator: {e}")
                st.warning("Falling back to SimpleEvaluator due to an error in Custom Evaluator setup.")
                return SimpleEvaluator()
        elif evaluator_type == EvaluatorType.SIMPLE.value:
            return SimpleEvaluator()
        elif evaluator_type == EvaluatorType.DEEP_EVAL.value:
            return DeepEval(**cfg.get(constants.CONFIG_PARAM, {}))
        else:
            st.warning(f"Unknown or unset evaluator type: {evaluator_type}. Defaulting to SimpleEvaluator.")
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
            from RAG_App.infrastructure.document_loaders.pdf_loader import PDFLoader
            from RAG_App.infrastructure.document_loaders.docx_loader import DOCXLoader
            from RAG_App.infrastructure.document_loaders.txt_loader import TXTLoader
            from RAG_App.infrastructure.document_loaders.csv_loader import CSVLoader
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
            st.error(f"Error extracting text: {e}, Traceback: {traceback.print_exc()}")
            return None, None
         
    # process documents
    def process_document(self, file, texts=None):
        try:
            print(self.chunker)
            chunks =  self.chunker.split_text(text=texts)

            print("Texts: ", chunks)
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
            print("Self Vector Store: ", self.vector_store)
            self.vector_store.add_embeddings(embeddings, documents)

            return documents, chunks
        except Exception as e:
            st.error(f"Error processing document: {e}, Traceback: {traceback.print_exc()}")
            return None, None

    def rewrite_query(self, query_text, max_assistant_chars=100):
        if not st.session_state.messages:
            return f"The user is asking: {query_text}"

        prev_user, prev_assistant = st.session_state.messages[-1]
        assistant_trimmed = prev_assistant.strip().replace('\n', ' ')[:max_assistant_chars] + "..."
        
        summary = (
            f"The user previously asked:\n{prev_user}\n"
            f"I responded with:\n{assistant_trimmed}\n"
            f"Now the user wants:\n{query_text}"
        )
        return summary
    
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
    
    def query(self, query_text, top_k=None):
        try:
            if self.query_classifier.is_greeting(query_text):
                st.markdown(self.query_classifier.get_greeting_response())
                return {
                constants.ANSWER: self.query_classifier.get_greeting_response(),
                constants.CONTEXTS: "",
                constants.RERANK_EXPLANATION: ""
            }
            # query_text = self.rewrite_query(query_text)
            # Ensure documents are available
            
            context_docs, explanation, context_docs_list = self.get_context_docs(query_text)
            print("CcontextDocsLen: ", len(context_docs))
            if self.query_classifier.is_irrelevant(query_text, context_docs):
                st.markdown(self.query_classifier.get_irrelevant_question_response())
                return {
                constants.ANSWER: self.query_classifier.get_irrelevant_question_response(),
                constants.CONTEXTS: context_docs,
                constants.RERANK_EXPLANATION: ""
            }
            # Join contexts
            context = "\n\n".join(context_docs)
            history_text = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in st.session_state.messages])
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
            answer_placeholder = st.empty()
            full_response = ""
            for delta in self.llm_service.generate_response(answer_prompt):
                full_response += delta
                answer_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
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
            st.error(f"Error during query: {e}, Traceback: {traceback.print_exc()}")
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
            st.error(f"Error during evaluation: {e}, Traceback: {traceback.print_exc()}")
            return None
