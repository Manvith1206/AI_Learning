import streamlit as st
from typing import List, Tuple, Dict, Any, Callable


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
    def initialize_session_state(variables: Dict[str, Any]):
        """Initialize session state variables if they don't exist"""
        for var_name, default_value in variables.items():
            if var_name not in st.session_state:
                st.session_state[var_name] = default_value
