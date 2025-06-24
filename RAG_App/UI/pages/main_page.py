from typing import Dict, List, Any, Optional

from UI.ui_components import UIComponents
from UI.chat_interface import ChatInterface
from UI.sidebar import Sidebar
from UI.metrics_display import MetricsDisplay
from UI.flashcard_display import FlashcardDisplay
from models import Flashcard
from infrastructure.common.rag_pipeline import RAGPipeline
from config import ConfigManager
from infrastructure.common import rag_constants as constants
from Utils.exceptions import FlashcardGenerationError, PipelineError

class MainPage:
    """Main application page integrating all components"""
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        config_manager: ConfigManager
    ):
        """Initialize the main page with all required use cases"""
        self.pipeline = pipeline
        self.config_manager = config_manager
        self.initialize_components()
        
    
    def initialize_components(self):
        """Initialize all UI components"""
        # UIComponents.initialize_page()
        # Chat interface
        self.chat_interface = ChatInterface(
            on_message_callback=None,
        )
        
        # Sidebar component
        self.sidebar = Sidebar(
        )
        self.flashcard_display = FlashcardDisplay(
            flashcards=[] # Initially empty, will be populated from session state
        )
       
    def trigger_flashcard_generation(self):
        """Handles the flashcard generation process using RAGPipeline."""
        if not self.pipeline or not hasattr(self.pipeline, 'generate_flashcards_from_text'):
            UIComponents.display_error("RAG Pipeline not available or misconfigured for flashcard generation.")
            UIComponents.set_session_state_variable("flashcards", [])
            UIComponents.set_session_state_variable("flashcards_generation_attempted", True)
            return

        # Attempt to get processed document content. 
        # This assumes 'processed_document_texts' is a list of strings (chunks/documents)
        # stored in session state after document processing via the sidebar/upload logic.
        # You might need to adjust this based on how your application stores processed text.
        docs_for_flashcards_content = UIComponents.get_session_state_variable("processed_document_texts", []) 

        if not docs_for_flashcards_content:
            UIComponents.display_warning("No processed document content available for flashcard generation. Please ensure documents are uploaded and processed first.")
        
        # Concatenate all document/chunk texts into a single string
        # Ensure docs_for_flashcards_content is a list of strings
        if isinstance(docs_for_flashcards_content, list) and all(isinstance(item, str) for item in docs_for_flashcards_content):
            full_text_content = "\n\n---\n\n".join(docs_for_flashcards_content)
        elif isinstance(docs_for_flashcards_content, str): # If it's already a single string
            full_text_content = docs_for_flashcards_content
        else:
            UIComponents.display_error("Document content for flashcards is in an unexpected format.")
            return

        if not full_text_content.strip():
            UIComponents.display_warning("No text content available to generate flashcards from.")
            UIComponents.set_session_state_variable("flashcards", [])
            UIComponents.set_session_state_variable("flashcards_generation_attempted", True)
            return

        generated_flashcards = []  # Initialize to ensure it's defined
        try:
            with UIComponents.display_spinner("Generating flashcards..."):
                UIComponents.display_info("Generating flashcards... This may take a moment.")
                generated_flashcards = self.pipeline.flashcards_generation.generate_flashcards_from_text(
                    full_text_content, 
                    num_flashcards=constants.NUM_OF_FLASHCARDS
                )
        except FlashcardGenerationError as e:
            UIComponents.display_error(f"Could not generate flashcards: {e}")
            # generated_flashcards remains []
        except PipelineError as e:
            UIComponents.display_error(f"A pipeline error occurred during flashcard generation: {e}")
            # generated_flashcards remains []
        except Exception as e:
            # In a production app, you might want to log this error in more detail
            UIComponents.display_error(f"An unexpected error occurred while generating flashcards: {e}")
            # generated_flashcards remains []
        
        UIComponents.set_session_state_variable("flashcards", generated_flashcards)
        UIComponents.set_session_state_variable("flashcards_generation_attempted", True)
        UIComponents.set_session_state_variable("card_index", 0)  # Reset index for new flashcards
        UIComponents.set_session_state_variable("show_answer", False)  # Reset answer visibility
        
        # Display success or warning based on generation result
        if generated_flashcards:
            UIComponents.display_success(f"{len(generated_flashcards)} flashcards generated successfully!")
        else:
            UIComponents.display_warning("No flashcards were generated by the LLM. The content might have been unsuitable or an error occurred.")

    def render(self):
        """Render the main page"""        
        # Render page title
        UIComponents.create_title("RAG Modular Application")
        
        # Create main tabs
        tabs = UIComponents.create_tabs(["Chat", "Evaluation", "FlashCards", "Debug", "Upload File"])
        
        # Chat tab
        with tabs[0]:
            self.chat_interface.render()
        
        # Evaluation tab
        with tabs[1]:
            self.sidebar.render_evaluation_section()
            self.sidebar.render_evaluation_config()
            self.render_test_all_configs()
        
        # Flashcards tab
        with tabs[2]:
            UIComponents.create_subheader_UI("Flashcard Generation & Review")

            if UIComponents.create_button("✨ Generate Flashcards"): 
                self.trigger_flashcard_generation()

            # Ensure session state keys exist before first use
            if "flashcards" not in UIComponents.get_session_state():
                 UIComponents.set_session_state_variable("flashcards", [])
            if "flashcards_generation_attempted" not in UIComponents.get_session_state():
                 UIComponents.set_session_state_variable("flashcards_generation_attempted", False)

            flashcards = UIComponents.get_session_state_variable("flashcards", [])
            flashcards_generation_attempted = UIComponents.get_session_state_variable("flashcards_generation_attempted", False)

            if not flashcards:
                if flashcards_generation_attempted:
                    UIComponents.display_info("No flashcards were generated. Try again or check your document source if applicable.")
                else:
                    UIComponents.display_info("Click 'Generate Flashcards' to create new flashcards.")
            else:
                # Update the existing flashcard_display instance with new/retrieved flashcards
                self.flashcard_display.flashcards = flashcards 
                # Initialize session state for card navigation (e.g., reset index if flashcards change)
                self.flashcard_display.initialze_session_state() 
                
                self.flashcard_display.display_card()
                self.flashcard_display.create_columns()
                self.flashcard_display.show_or_hide_answer()
        
        # Debug tab
        with tabs[3]:
            self.render_debug_tab()
        # Upload File Section
        with tabs[4]:
            self.render_upload_file_section()
    
    def render_upload_file_section(self):
        """Render file upload section in sidebar with caching."""
        UIComponents.create_subheader_UI("Upload Documents")
        uploaded_file = UIComponents.create_file_uploader(
            "Upload a document to start",
            file_types=["pdf", "docx", "txt", "csv"]
        )

        self.load_pre_processed_docs_or_process_the_doc(uploaded_file)

    def render_test_all_configs(self):
        from infrastructure.testing.RAG_Testing import test_rag_combinations
        if UIComponents.create_button("Test All Configurations", key="TEST_ALL_CONFIGS"):
            test_rag_combinations()

    from streamlit.runtime.uploaded_file_manager import UploadedFile
    def load_pre_processed_docs_or_process_the_doc(self, uploaded_file: UploadedFile):
        """
        Process the uploaded document by retrieving a pre-processed version from cache if available,
        or by extracting text and processing the document to generate a new vector store if not.
        Parameters:
            uploaded_file (file-like object): The uploaded document to be processed. It should support
                                               a 'getvalue()' method for retrieving file content, and
                                               must have a 'name' attribute representing the file name.
        Behavior:
            - Retrieves the current configuration parameters for the chunker, embedder, and vector store.
            - Generates a unique cache key based on the document's content and processing configurations.
            - If a cached vector store exists for the generated key, it loads the vector store and updates
              the session state with the associated documents and their texts.
            - If no cached vector store is found, it extracts text from the document, processes the document
              to generate a new vector store, caches this result, and updates the session state accordingly.
            - Provides user feedback for successful operations or errors encountered during processing via UI components.
        Raises:
            Exception: Any exception that occurs during text extraction or document processing is caught and
                       reported using UIComponents error messages rather than being propagated.
        Returns:
            None: The method updates relevant session state variables and UI components based on the processing outcome.
        """
        import Utils.utils

        if uploaded_file:
            pipeline = Utils.utils.get_pipeline()
            vector_store = pipeline.component_manager.get_vector_store()

            # If vector_store already has documents, assume it's loaded from cache/persistence
            if vector_store and vector_store.documents:
                UIComponents.display_success(f"Loaded pre-processed document '{uploaded_file.name}' from persistent store.")
                # Ensure session state is also aligned
                UIComponents.set_session_state_variable("documents", vector_store.documents)
                UIComponents.set_session_state_variable("processed_document_texts", [doc.page_content for doc in vector_store.documents])
                return

            # If not loaded, process the document
            with UIComponents.display_spinner(f"Processing document: {uploaded_file.name}..."):
                try:
                    # Extract text from the uploaded file
                    text_content = self.pipeline.text_extractor.extract_text(uploaded_file)
                    
                    if text_content:
                        # Process the document to update the vector store
                        processed_vector_store = self.pipeline.process_document(text_content)
                        
                        if processed_vector_store:
                            # The pipeline's vector_store is already updated by process_document
                            UIComponents.set_session_state_variable("vector_store", processed_vector_store)
                            documents = processed_vector_store.documents
                            UIComponents.set_session_state_variable("documents", documents)
                            if documents:
                                UIComponents.set_session_state_variable("processed_document_texts", [doc.page_content for doc in documents])
                                
                            UIComponents.display_success(f"Document '{uploaded_file.name}' processed and saved to persistent store.")
                        else:
                            UIComponents.display_error("Failed to process the document.")
                    else:
                        UIComponents.display_error("Failed to extract text from the document.")
                except Exception as e:
                    UIComponents.display_error(f"An error occurred during processing: {e}")

    def render_debug_tab(self):
        """Render the debug tab with system information"""
        UIComponents.create_subheader_UI("Debug Information")
        
        with UIComponents.create_expander("Session State", expanded=False):
            UIComponents.write(UIComponents.get_session_state())
        UIComponents.write("Component Metrics")
        UIComponents.write("This section provides the performance and cost metrics for each step in the RAG pipeline.")
        UIComponents.write("The metrics include time taken for processing and estimated cost for each step.")
        UIComponents.write("**Note:** The cost is an estimate based on the current configuration and may vary based on actual usage.")
        if hasattr(UIComponents.get_session_state(), "documents") and UIComponents.get_session_state_variable("documents", None):
            metrics = self.get_component_metrics()
            MetricsDisplay.display_pipeline_metrics(metrics)
        else:
            UIComponents.display_info("Process documents to view component metrics")
    
    # --- Component type getters ---
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        return self.config_manager.get_config(component_name)
    
    def get_component_metrics(self) -> Dict[str, Any]:
        """Get metrics for all components"""
        # This would be implemented to retrieve metrics from the RAG service
        # For now, return mock data
        if not self.pipeline:
            UIComponents.display_error("RAG Pipeline not available for metrics.")
            return {}
        return {
            constants.CONFIG_CHUNKER: self.pipeline.component_manager.get_chunker_cost_and_time(),
            constants.CONFIG_EMBEDDER: self.pipeline.component_manager.get_embedder_cost_and_time(),
            constants.CONFIG_RETRIEVER: self.pipeline.component_manager.get_retriever_cost_and_time(),
            constants.CONFIG_RERANKER: self.pipeline.component_manager.get_reranker_cost_and_time(),
            constants.CONFIG_EVALUATOR: self.pipeline.component_manager.get_evaluator_cost_and_time(),
            constants.CONFIG_VECTOR_STORE: self.pipeline.component_manager.get_vector_store_cost_and_time(),
            constants.CONFIG_LLM: self.pipeline.component_manager.get_llm_service_cost_and_time()
        }