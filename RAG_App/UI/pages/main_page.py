from typing import Dict, List, Any, Optional

from UI.UI_Components import UIComponents
from UI.chat_interface import ChatInterface
from UI.sidebar import Sidebar
from UI.metrics_display import MetricsDisplay
from UI.flashcard_display import FlashcardDisplay
from models import Flashcard, AppConfig
from infrastructure.common.rag_pipeline import RAGPipeline
from infrastructure.common import RAG_Constants as constants
from infrastructure.common.exceptions import FlashcardGenerationError, PipelineError
import tempfile
import os

class MainPage:
    """Main application page integrating all components"""
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        config: AppConfig
    ):
        """Initialize the main page with all required use cases"""
        self.pipeline = pipeline
        self.config = config
        self.initialize_components()
        
    
    def initialize_components(self):
        """Initialize all UI components"""
        self.chat_interface = ChatInterface(pipeline=self.pipeline)
        self.sidebar = Sidebar()
        self.flashcard_display = FlashcardDisplay(
            flashcards=[] # Initially empty, will be populated from session state
        )
       
    def trigger_flashcard_generation(self):
        """Handles the flashcard generation process using RAGPipeline."""
        if not self.pipeline or not hasattr(self.pipeline, 'flashcards_generation'):
            UIComponents.display_error("RAG Pipeline not available or misconfigured for flashcard generation.")
            UIComponents.set_session_state_variable("flashcards", [])
            UIComponents.set_session_state_variable("flashcards_generation_attempted", True)
            return

        docs_for_flashcards_content = UIComponents.get_session_state_variable("processed_document_texts", []) 

        if not docs_for_flashcards_content:
            UIComponents.display_warning("No processed document content available for flashcard generation. Please ensure documents are uploaded and processed first.")
            return
        
        if isinstance(docs_for_flashcards_content, list) and all(isinstance(item, str) for item in docs_for_flashcards_content):
            full_text_content = "\n\n---\n\n".join(docs_for_flashcards_content)
        elif isinstance(docs_for_flashcards_content, str):
            full_text_content = docs_for_flashcards_content
        else:
            UIComponents.display_error("Document content for flashcards is in an unexpected format.")
            return

        if not full_text_content.strip():
            UIComponents.display_warning("No text content available to generate flashcards from.")
            UIComponents.set_session_state_variable("flashcards", [])
            UIComponents.set_session_state_variable("flashcards_generation_attempted", True)
            return

        generated_flashcards = []
        try:
            with UIComponents.display_spinner("Generating flashcards..."):
                generated_flashcards = self.pipeline.flashcards_generation.generate_flashcards_from_text(
                    full_text_content, 
                    num_flashcards=constants.Constants.NUM_OF_FLASHCARDS
                )
                if generated_flashcards:
                    UIComponents.display_success("Flashcards generated successfully!")
                else:
                    UIComponents.display_warning("Could not generate flashcards from the provided text.")
        except FlashcardGenerationError as e:
            UIComponents.display_error(f"Flashcard Generation Error: {e}")
        except Exception as e:
            UIComponents.display_error(f"An unexpected error occurred during flashcard generation: {e}")
        finally:
            UIComponents.set_session_state_variable("flashcards", generated_flashcards)
            UIComponents.set_session_state_variable("flashcards_generation_attempted", True)
            
    def render(self):
        """Render the main page"""
        UIComponents.create_title("RAG-Bot")
        UIComponents.set_page_container_style()
        
        with self.sidebar:
            self.render_upload_file_section()
            self.sidebar.render_sidebar()
            
        
        chat_tab, flashcards_tab, metrics_tab, debug_tab = UIComponents.create_tabs(
            ["Chat", "FlashCards", "Metrics", "Debug"]
        )

        with chat_tab:
            self.chat_interface.render()
        
        with flashcards_tab:
            if UIComponents.create_button("Generate Flashcards"):
                self.trigger_flashcard_generation()
            
            flashcards_to_display = UIComponents.get_session_state_variable("flashcards", [])
            if UIComponents.get_session_state_variable("flashcards_generation_attempted", False) and not flashcards_to_display:
                UIComponents.display_info("No flashcards were generated. Try different content or check logs.")
            
            self.flashcard_display.flashcards = flashcards_to_display

        with metrics_tab:
            UIComponents.create_subheader_UI("Component Metrics")
            UIComponents.write("Performance and cost metrics for each step in the RAG pipeline.")
            if UIComponents.get_session_state_variable("documents", None):
                metrics = self.get_component_metrics()
                MetricsDisplay.display_pipeline_metrics(metrics)
            else:
                UIComponents.display_info("Process a document to view component metrics.")
        
        with debug_tab:
            self.render_debug_tab()

    def render_upload_file_section(self):
        """Render file upload section in sidebar with caching."""
        uploaded_file = UIComponents.create_file_uploader(
            label="Upload your document",
            type=['pdf', 'txt', 'docx']
        )
        if uploaded_file:
            self.load_pre_processed_docs_or_process_the_doc(uploaded_file)

    def load_pre_processed_docs_or_process_the_doc(self, uploaded_file: 'UploadedFile'):
        if uploaded_file is None:
            return
        with UIComponents.display_spinner(f"Processing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            try:
                from infrastructure.utils.text_extractor import TextExtractorFactory
                text_extractor = TextExtractorFactory.get_extractor(file_path)
                if not text_extractor:
                    UIComponents.display_error("Failed to find a suitable text extractor for the uploaded file type.")
                    return
                texts = text_extractor.extract_text()
                if not texts:
                    UIComponents.display_error("Failed to extract text from the document.")
                    return
                
                processed_vector_store = self.pipeline.process_document(file_path=file_path, texts=texts)
                
                if processed_vector_store:
                    UIComponents.set_session_state_variable("vector_store", processed_vector_store)
                    documents = processed_vector_store.documents
                    UIComponents.set_session_state_variable("documents", documents)
                    if documents:
                        UIComponents.set_session_state_variable("processed_document_texts", [doc.page_content for doc in documents])
                    UIComponents.display_success(f"Document '{uploaded_file.name}' processed successfully.")
                else:
                    UIComponents.display_error("Failed to process the document.")
            except PipelineError as e:
                UIComponents.display_error(f"A pipeline error occurred: {e}")
            except Exception as e:
                UIComponents.display_error(f"An unexpected error occurred during processing: {e}")
            finally:
                os.remove(file_path)

    def render_debug_tab(self):
        """Render the debug tab with system information"""
        UIComponents.create_subheader_UI("Debug Information")
        with UIComponents.create_expander("Session State", expanded=False):
            UIComponents.write(UIComponents.get_session_state())
        UIComponents.write("Component Metrics")
        if UIComponents.get_session_state_variable("documents", None):
            metrics = self.get_component_metrics()
            MetricsDisplay.display_pipeline_metrics(metrics)
        else:
            UIComponents.display_info("Process a document to view component metrics")
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        if hasattr(self.config, component_name):
            component_config = getattr(self.config, component_name)
            if hasattr(component_config, 'dict'):
                return component_config.dict()
        return {}
    
    def get_component_metrics(self) -> Dict[str, Any]:
        """Get metrics for all components"""
        if not self.pipeline or not self.pipeline.components:
            UIComponents.display_error("RAG Pipeline not available for metrics.")
            return {}
        
        metrics = {}
        component_map = {
            constants.ConfigManagerNames.CONFIG_CHUNKER: constants.ConfigManagerNames.CONFIG_CHUNKER,
            constants.ConfigManagerNames.CONFIG_EMBEDDER: constants.ConfigManagerNames.CONFIG_EMBEDDER,
            constants.ConfigManagerNames.CONFIG_RETRIEVER: constants.ConfigManagerNames.CONFIG_RETRIEVER,
            constants.ConfigManagerNames.CONFIG_RERANKER: constants.ConfigManagerNames.CONFIG_RERANKER,
            constants.ConfigManagerNames.CONFIG_EVALUATOR: constants.ConfigManagerNames.CONFIG_EVALUATOR,
            constants.ConfigManagerNames.CONFIG_VECTOR_STORE: constants.ConfigManagerNames.CONFIG_VECTOR_STORE,
            constants.ConfigManagerNames.CONFIG_LLM: constants.ConfigManagerNames.CONFIG_LLM,
        }

        for metric_name, component_key in component_map.items():
            component_instance = self.pipeline.components.get(component_key)
            if component_instance and hasattr(component_instance, 'get_cost_and_time_taken'):
                metrics[metric_name] = component_instance.get_cost_and_time_taken()
            else:
                metrics[metric_name] = (0, 0)  # Default value
                
        return metrics