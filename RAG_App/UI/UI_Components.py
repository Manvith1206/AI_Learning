import streamlit as st
from typing import List, Tuple, Dict, Any, Callable
import pandas as pd
from RAG_App.infrastructure.Common.rag_pipeline import RAGPipeline

class UIComponents:
    """Base UI components for the RAG application"""
    
    @staticmethod
    def initialize_page():
        """Initialize the Streamlit page configuration"""
        st.set_page_config(
            page_title="RAG Modular",
            page_icon=":notebook:",
            layout="wide"
        )
    
    @staticmethod
    def create_tabs(tab_names: List[str]):
        """Create tabs with the given names"""
        return st.tabs(tab_names)
    
    @staticmethod
    def create_sidebar_tabs(tab_names: List[str]):
        """Create tabs in the sidebar"""
        with st.sidebar:
            return st.tabs(tab_names)
    @staticmethod
    def create_sidebar():
        """Create tabs in the sidebar"""
        with st.sidebar:
            pass  # Empty sidebar
        return st.sidebar
    
    @staticmethod
    def display_success(message: str):
        """Display a success message"""
        st.success(message)
    
    @staticmethod
    def display_error(message: str):
        """Display an error message"""
        st.error(message)
    
    @staticmethod
    def display_warning(message: str):
        """Display a warning message"""
        st.warning(message)
    
    @staticmethod
    def display_info(message: str):
        """Display an info message"""
        st.info(message)
    
    @staticmethod
    def display_spinner(message: str):
        """Create a spinner context manager"""
        return st.spinner(message)
    
    @staticmethod
    def create_expander(title: str, expanded: bool = False):
        """Create an expander"""
        return st.expander(title, expanded=expanded)
    
    @staticmethod
    def create_columns(num_columns: int):
        """Create columns"""
        return st.columns(num_columns)
    
    @staticmethod
    def display_metric(label: str, value: Any):
        """Display a metric"""
        st.metric(label=label, value=value)
    
    @staticmethod
    def display_divider():
        """Display a divider"""
        st.divider()
    
    @staticmethod
    def display_header(text: str, level: int = 1):
        """Display a header with the specified level"""
        if level == 1:
            st.title(text)
        elif level == 2:
            st.header(text)
        elif level == 3:
            st.subheader(text)
        else:
            st.write(f"**{text}**")
    
    @staticmethod
    def create_subheader(text: str):
        """Create a subheader"""
        st.subheader(text)          

    @staticmethod
    def write(text: str):
        """Write text to the Streamlit app"""
        st.write(text)

    @staticmethod
    def initialize_session_state(variables: Dict[str, Any]):
        """Initialize session state variables if they don't exist"""
        for var_name, default_value in variables.items():
            if var_name not in st.session_state:
                st.session_state[var_name] = default_value

    import RAG_App.config as ConfigManager
    @staticmethod
    def initialize_pipeline(config_manager: ConfigManager):
        """Initialize the RAG pipeline in session state"""
        if "pipeline" not in st.session_state:
            st.session_state.pipeline_config = config_manager
            st.session_state.pipeline_created = False

    @staticmethod
    def get_session_state_variable(var_name: str, default_value: Any = None) -> Any:
        """Get a session state variable, initializing it if it doesn't exist"""
        if var_name not in st.session_state:
            st.session_state[var_name] = default_value
        return st.session_state[var_name]
    
    @staticmethod
    def set_session_state_variable(var_name: str, value: Any):
        """Set a session state variable"""
        st.session_state[var_name] = value

    @staticmethod
    def get_session_state_messages() -> List[Dict[str, Any]]:
        """Get the chat messages from session state"""
        return st.session_state.messages
    
    @staticmethod
    def display_message_with_role(self, role: str, message: str):
        """Display a message with a specific role in the chat"""
        with st.chat_message(role):
            st.markdown(message)
    @staticmethod
    def create_button(label: str, key: str = None):
        """Create a button with an optional key"""
        return st.button(label, key=key)
    
    @staticmethod
    def create_file_uploader(label: str, file_types: List[str], accept_multiple_files: bool = False):
        """Create a file uploader with specified file types"""
        return st.file_uploader(label, type=file_types, accept_multiple_files=accept_multiple_files)
    @staticmethod
    def create_text_area(label: str, key: str = None, value: str = ""):
        """Create a text input field"""
        return st.text_area(label, value=value, key=key)
    @staticmethod
    def add_message_to_chat(self, role: str, content: str):
        """Add a message to the chat history"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": role, "content": content})

    @staticmethod
    def chat_input(label: str):
        """Create a chat input field"""
        st.chat_input(label)
    @staticmethod 
    def process_chat_input(role: str, content: str, pipeline: RAGPipeline, prompt: str):
        """Display a chat message with a specific role"""
        with st.chat_message(role):
            with st.spinner("🤔 Thinking..."):
                response = pipeline.query(prompt)
                st.markdown(f"**Re-ranking Explanation:**\n{response['rerank_explanation']}")
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": role, "content": response["answer"]})

    @staticmethod
    def markdown(text: str):
        """Display markdown text"""
        st.markdown(text)

    @staticmethod
    def selectbox(label: str, options: List[str], index: int = 0):
        """Create a selectbox"""
        return st.selectbox(label, options=options, index=index)
    @staticmethod
    def display_dataframe(df: pd.DataFrame):
        """Display a DataFrame"""
        st.dataframe(df)
    @staticmethod
    def display_bar_chart(values):
        """Display a bar chart from a DataFrame"""
        st.bar_chart(values)
    @staticmethod
    def display_slider(label: str, min_value: int, max_value: int, value: int = None, step: int = 1):
        """Create a slider"""
        return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step)
    @staticmethod
    def create_number_input(label: str, min_value: int, max_value: int, value: int = None):
        """Create a number input field"""
        return st.number_input(label, min_value=min_value, max_value=max_value, value=value)