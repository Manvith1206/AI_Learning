import streamlit as st
from typing import Dict, List, Any, Optional

from app.presentation.components.ui_components import UIComponents
from app.presentation.components.chat_interface import ChatInterface
from app.presentation.components.sidebar import Sidebar
from app.presentation.components.metrics_display import MetricsDisplay
from app.presentation.components.flashcard_display import FlashcardDisplay

from app.application.usecases.document_processing import DocumentProcessingUseCase
from app.application.usecases.chat_generation import ChatGenerationUseCase
from app.application.usecases.flashcard_generation import FlashcardGenerationUseCase
from app.application.usecases.evaluation import EvaluationUseCase
from app.application.usecases.configuration import ConfigurationUseCase


class MainPage:
    """Main application page integrating all components"""
    
    def __init__(
        self,
        document_processing_usecase: DocumentProcessingUseCase,
        chat_generation_usecase: ChatGenerationUseCase,
        flashcard_generation_usecase: FlashcardGenerationUseCase,
        evaluation_usecase: EvaluationUseCase,
        configuration_usecase: ConfigurationUseCase
    ):
        """Initialize the main page with all required use cases"""
        self.document_processing_usecase = document_processing_usecase
        self.chat_generation_usecase = chat_generation_usecase
        self.flashcard_generation_usecase = flashcard_generation_usecase
        self.evaluation_usecase = evaluation_usecase
        self.configuration_usecase = configuration_usecase
        
        # Initialize UI components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all UI components"""
        # Chat interface
        self.chat_interface = ChatInterface(
            on_message_callback=self._handle_chat_message
        )
        
        # Sidebar component
        self.sidebar = Sidebar(
            on_document_upload=self._handle_document_upload,
            on_chunker_config_update=self._handle_chunker_config_update,
            on_embedder_config_update=self._handle_embedder_config_update,
            on_vector_store_config_update=self._handle_vector_store_config_update,
            on_retriever_config_update=self._handle_retriever_config_update,
            on_reranker_config_update=self._handle_reranker_config_update,
            on_llm_config_update=self._handle_llm_config_update,
            on_evaluator_config_update=self._handle_evaluator_config_update,
            get_chunker_types=self._get_chunker_types,
            get_embedder_types=self._get_embedder_types,
            get_vector_store_types=self._get_vector_store_types,
            get_retriever_types=self._get_retriever_types,
            get_reranker_types=self._get_reranker_types,
            get_llm_types=self._get_llm_types,
            get_evaluator_types=self._get_evaluator_types,
            get_current_config=self._get_component_config
        )
        
        # Flashcard display
        self.flashcard_display = FlashcardDisplay(
            on_generate_flashcards=self._handle_flashcard_generation
        )
    
    def render(self):
        """Render the main page"""
        UIComponents.initialize_page()
        
        # Render page title
        st.title("RAG Modular Application")
        
        # Render sidebar
        self.sidebar.render()
        
        # Create main tabs
        tabs = UIComponents.create_tabs(["Chat", "Evaluation", "FlashCards", "Debug"])
        
        # Chat tab
        with tabs[0]:
            self.chat_interface.render()
        
        # Evaluation tab
        with tabs[1]:
            MetricsDisplay.display_evaluation_section(
                on_evaluate_callback=self._handle_evaluation
            )
        
        # Flashcards tab
        with tabs[2]:
            self.flashcard_display.render()
        
        # Debug tab
        with tabs[3]:
            self._render_debug_tab()
    
    def _render_debug_tab(self):
        """Render the debug tab with system information"""
        st.subheader("Debug Information")
        
        with st.expander("Session State", expanded=False):
            st.write(st.session_state)
        
        with st.expander("Component Metrics", expanded=False):
            if hasattr(st.session_state, "documents") and st.session_state.documents:
                metrics = self._get_component_metrics()
                MetricsDisplay.display_pipeline_metrics(metrics)
            else:
                st.info("Process documents to view component metrics")
    
    # --- Callback handlers ---
    
    def _handle_document_upload(self, uploaded_file) -> tuple:
        """Handle document upload and processing"""
        try:
            document = self.document_processing_usecase.extract_text_from_file(uploaded_file)
            chunks = self.document_processing_usecase.process_document(document)
            return document, chunks
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return None, None
    
    def _handle_chat_message(self, query: str) -> Dict[str, Any]:
        """Handle chat message and generate response"""
        try:
            result = self.chat_generation_usecase.generate_response(query)
            return {
                "answer": result.answer,
                "rerank_explanation": result.rerank_explanation,
                "retrieved_documents": result.retrieved_documents
            }
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return {"answer": f"Error: {str(e)}", "rerank_explanation": ""}
    
    def _handle_flashcard_generation(self):
        """Handle flashcard generation"""
        try:
            if st.session_state.documents:
                document = st.session_state.documents
                flashcards = self.flashcard_generation_usecase.generate_flashcards(document)
                return flashcards
            return []
        except Exception as e:
            st.error(f"Error generating flashcards: {str(e)}")
            return []
    
    def _handle_evaluation(self, ground_truth: str) -> Dict[str, float]:
        """Handle evaluation of last query"""
        try:
            evaluation_result = self.evaluation_usecase.evaluate_last_query(ground_truth)
            return evaluation_result.metrics
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            return {}
    
    # --- Configuration handlers ---
    
    def _handle_chunker_config_update(self, config: Dict[str, Any]):
        """Handle chunker configuration update"""
        self.configuration_usecase.update_component_config("chunker", config)
    
    def _handle_embedder_config_update(self, config: Dict[str, Any]):
        """Handle embedder configuration update"""
        self.configuration_usecase.update_component_config("embedder", config)
    
    def _handle_vector_store_config_update(self, config: Dict[str, Any]):
        """Handle vector store configuration update"""
        self.configuration_usecase.update_component_config("vector_store", config)
    
    def _handle_retriever_config_update(self, config: Dict[str, Any]):
        """Handle retriever configuration update"""
        self.configuration_usecase.update_component_config("retriever", config)
    
    def _handle_reranker_config_update(self, config: Dict[str, Any]):
        """Handle reranker configuration update"""
        self.configuration_usecase.update_component_config("reranker", config)
    
    def _handle_llm_config_update(self, config: Dict[str, Any]):
        """Handle LLM configuration update"""
        self.configuration_usecase.update_component_config("llm", config)
    
    def _handle_evaluator_config_update(self, config: Dict[str, Any]):
        """Handle evaluator configuration update"""
        self.configuration_usecase.update_component_config("evaluator", config)
    
    # --- Component type getters ---
    
    def _get_chunker_types(self) -> List[str]:
        """Get available chunker types"""
        return ["recursive", "semantic", "sentence"]
    
    def _get_embedder_types(self) -> List[str]:
        """Get available embedder types"""
        return ["openai", "cohere", "tfidf", "gemini"]
    
    def _get_vector_store_types(self) -> List[str]:
        """Get available vector store types"""
        return ["chroma", "faiss", "scikit_learn", "pinecone"]
    
    def _get_retriever_types(self) -> List[str]:
        """Get available retriever types"""
        return ["similarity", "mmr", "hybrid"]
    
    def _get_reranker_types(self) -> List[str]:
        """Get available reranker types"""
        return ["none", "llm", "cohere", "jina"]
    
    def _get_llm_types(self) -> List[str]:
        """Get available LLM types"""
        return ["openai", "anthropic", "gemini", "mock"]
    
    def _get_evaluator_types(self) -> List[str]:
        """Get available evaluator types"""
        return ["llm", "ragas", "none"]
    
    def _get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        return self.configuration_usecase.get_component_config(component_name)
    
    def _get_component_metrics(self) -> Dict[str, Any]:
        """Get metrics for all components"""
        # This would be implemented to retrieve metrics from the RAG service
        # For now, return mock data
        return {
            "Chunking": ("$0.0001", "0.12s"),
            "Embedding": ("$0.0015", "0.45s"),
            "Retrieval": ("$0.0003", "0.22s"),
            "Reranking": ("$0.0020", "0.65s"),
            "LLM Generation": ("$0.0120", "1.35s")
        }
