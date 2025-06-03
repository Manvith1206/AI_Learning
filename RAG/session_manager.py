import streamlit as st
from rag_modular.Common.config_manager import ConfigManager
import rag_modular.Common.RAG_Constants as constants
from rag_modular.Common.RAG_Constants import GeminiLLMModel

def initialize_session_state():
    """Initialize all session state variables"""
    if "pipeline_config" not in st.session_state: # Changed from "pipeline" to "pipeline_config" for clarity
        config_manager = ConfigManager()
        st.session_state.pipeline_config = config_manager
    
    if "pipeline_created" not in st.session_state: # Explicitly track pipeline creation
        st.session_state.pipeline_created = False
    
    if "pipeline_instance" not in st.session_state: # To store the actual pipeline object
        st.session_state.pipeline_instance = None

    if "documents" not in st.session_state:
        st.session_state.documents = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "LLM_Model_Options" not in st.session_state:
        st.session_state.LLM_Model_Options = [e.value for e in GeminiLLMModel]
    if "LLM_Service" not in st.session_state:
        st.session_state.LLM_Service = constants.LLMServiceType.GEMINI.value
    
    # Add any other session state variables that might be needed globally
    # For example, selected configurations from the UI
    if "selected_chunker_config" not in st.session_state:
        st.session_state.selected_chunker_config = {} # Or default values
    if "selected_embedder_config" not in st.session_state:
        st.session_state.selected_embedder_config = {}
    # ... and so on for other configurable components
