from typing import Dict, List, Any, Optional

from UI.UI_Components import UIComponents
from UI.chat_interface import ChatInterface
from UI.sidebar import Sidebar
from UI.metrics_display import MetricsDisplay
from UI.flashcard_display import FlashcardDisplay
from models import Flashcard
from infrastructure.Common.rag_pipeline import RAGPipeline
from infrastructure.Common import RAG_Constants as constants
import Utils.Utils

class MainPage:
    """Main application page integrating all components"""
    
    def __init__(
        self
    ):
        """Initialize the main page with all required use cases"""
        
        # Initialize UI components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all UI components"""
        # Chat interface
        self.chat_interface = ChatInterface(
            on_message_callback=None
        )
        
        # Sidebar component
        self.sidebar = Sidebar(
        )
        self.flashcard_display = FlashcardDisplay(
            flashcards=self.get_flashcards()
        )
       
    def get_flashcards(self) -> List[FlashcardDisplay]:
        """Get the list of generated flashcards"""
        return [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    {"question": "What is the boiling point of water?", "answer": "100°C or 212°F"},
]
    #     return [
    #     FlashcardDisplay(
    #         id="fc001",
    #         question="What is the primary difference between asynchronous and synchronous transmission?",
    #         answer="Asynchronous transmission sends data one character at a time with start and stop bits, while synchronous transmission sends large blocks of data in a continuous stream without start and stop codes.",
    #         document_id="DCA2104 Unit-08_V1.1.pdf",
    #         document_chunk_id="sec2_async_vs_sync",
    #         metadata={"topic": "Transmission Methods", "page": 4}
    #     ),
    #     FlashcardDisplay(
    #         id="fc002",
    #         question="What are the two types of errors in digital transmission?",
    #         answer="Single-bit errors and burst errors.",
    #         document_id="DCA2104 Unit-08_V1.1.pdf",
    #         document_chunk_id="sec3_types_of_errors",
    #         metadata={"topic": "Error Types", "page": 8}
    #     ),
    #     FlashcardDisplay(
    #         id="fc003",
    #         question="Which technique is more efficient for large data blocks: asynchronous or synchronous transmission?",
    #         answer="Synchronous transmission is more efficient for large data blocks.",
    #         document_id="DCA2104 Unit-08_V1.1.pdf",
    #         document_chunk_id="sec2_async_vs_sync",
    #         metadata={"topic": "Efficiency Comparison", "page": 7}
    #     ),
    #     FlashcardDisplay(
    #         id="fc004",
    #         question="What are the three representations used to describe Cyclic Redundancy Check (CRC)?",
    #         answer="Modulo 2 arithmetic, polynomials, and digital logic.",
    #         document_id="DCA2104 Unit-08_V1.1.pdf",
    #         document_chunk_id="sec4_crc",
    #         metadata={"topic": "Error Detection", "page": 11}
    #     ),
    #     FlashcardDisplay(
    #         id="fc005",
    #         question="What is Hamming Code used for in data communication?",
    #         answer="It is used for single-bit error correction by identifying the bit position in error using parity bits.",
    #         document_id="DCA2104 Unit-08_V1.1.pdf",
    #         document_chunk_id="sec5_hamming_code",
    #         metadata={"topic": "Error Correction", "page": 16}
    #     ),
    #     FlashcardDisplay(
    #         id="fc006",
    #         question="What are the two modes of data transmission in line configuration?",
    #         answer="Full duplex and half duplex.",
    #         document_id="DCA2104 Unit-08_V1.1.pdf",
    #         document_chunk_id="sec6_line_config",
    #         metadata={"topic": "Line Configuration", "page": 20}
    #     ),
    # ]
    
    def render(self):
        """Render the main page"""
        # UIComponents.initialize_page()
        
        # Render page title
        UIComponents.create_title("RAG Modular Application")
        
        # Render sidebar
        self.sidebar.render_sidebar()
        
        # Create main tabs
        tabs = UIComponents.create_tabs(["Chat", "Evaluation", "FlashCards", "Debug"])
        
        # Chat tab
        with tabs[0]:
            self.chat_interface.render()
        
        # Evaluation tab
        with tabs[1]:
            # MetricsDisplay.display_evaluation_section(
            #     on_evaluate_callback=self._handle_evaluation
            # )
            pass
        
        # Flashcards tab
        with tabs[2]:
            self.flashcard_display.initialze_session_state()
            
            # Display current flashcard
            self.flashcard_display.display_card()
            
            # Create columns for navigation
            self.flashcard_display.create_columns()
            
            # Show or hide answer
            self.flashcard_display.show_or_hide_answer()
        
        # Debug tab
        with tabs[3]:
            self._render_debug_tab()
    
    def _render_debug_tab(self):
        """Render the debug tab with system information"""
        UIComponents.create_subheader_UI("Debug Information")
        
        with UIComponents.create_expander("Session State", expanded=False):
            UIComponents.write(UIComponents.get_session_state())
        
        UIComponents.write("Component Metrics")
        if hasattr(UIComponents.get_session_state(), "documents") and UIComponents.get_session_state_variable("documents", None):
            metrics = self._get_component_metrics()
            MetricsDisplay.display_pipeline_metrics(metrics)
        else:
            UIComponents.display_info("Process documents to view component metrics")
    
    # --- Component type getters ---
    def _get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        return self.configuration_usecase.get_component_config(component_name)
    
    def _get_component_metrics(self) -> Dict[str, Any]:
        """Get metrics for all components"""
        # This would be implemented to retrieve metrics from the RAG service
        # For now, return mock data
        pipeline = Utils.Utils.get_pipeline()
        return {
            constants.CONFIG_CHUNKER: pipeline.get_chunker_cost_and_time(),
            constants.CONFIG_EMBEDDER: pipeline.get_embedder_cost_and_time(),
            constants.CONFIG_RETRIEVER: pipeline.get_retriever_cost_and_time(),
            constants.CONFIG_RERANKER: pipeline.get_reranker_cost_and_time(),
            constants.CONFIG_EVALUATOR: pipeline.get_evaluator_cost_and_time(),
            constants.CONFIG_VECTOR_STORE: pipeline.get_vector_store_cost_and_time(),
            constants.CONFIG_LLM: pipeline.get_llm_service_cost_and_time()
        }
    
    # def _handle_evaluation(self, ground_truth: str) -> Dict[str, float]:
    #     """Handle evaluation of last query"""
    #     try:
    #         evaluation_result = self.evaluation_usecase.evaluate_last_query(ground_truth)
    #         return evaluation_result.metrics
    #     except Exception as e:
    #         UIComponents.display_error(f"Error during evaluation: {str(e)}")
    #         return {}