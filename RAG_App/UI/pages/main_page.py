import streamlit as st
from typing import Dict, List, Any, Optional

from RAG_App.UI.UI_Components import UIComponents
from RAG_App.UI.chat_interface import ChatInterface
from RAG_App.UI.sidebar import Sidebar
from RAG_App.UI.metrics_display import MetricsDisplay
from RAG_App.UI.flashcard_display import FlashcardDisplay

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
            on_message_callback=self._handle_chat_message
        )
        
        # Sidebar component
        self.sidebar = Sidebar(
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
    
    # --- Component type getters ---
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
