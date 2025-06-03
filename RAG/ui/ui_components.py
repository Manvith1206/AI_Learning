import streamlit as st
from rag_modular.Common.RAG_Constants import (
    TEXT_PROCESSING_DISPLAY_NAME, 
    RETRIEVAL_DISPLAY_NAME, 
    EVALUATION_DISPLAY_NAME
)

class UIComponents:
    @staticmethod
    def initialize_page():
        st.set_page_config(
            page_title="RAG Modular",
            page_icon=":notebook:",
            layout="wide"
        )
    
    @staticmethod
    def create_tabs():
        # We will return the actual tab objects to be used in main_app.py
        tab1, tab2, tab3 = st.tabs(["Chat with Documents", "Performance Metrics", "FlashCards"]) # Assuming FlashCards tab is still desired
        return tab1, tab2, tab3

    @staticmethod
    def render_config_tabs_in_sidebar(app_instance):
        """
        Renders configuration tabs within the sidebar.
        """
        with st.sidebar:
            st.subheader("Configuration")
            config_tabs = st.tabs([
                TEXT_PROCESSING_DISPLAY_NAME, 
                RETRIEVAL_DISPLAY_NAME, 
                EVALUATION_DISPLAY_NAME
            ])
        
            with config_tabs[0]:
                app_instance.render_text_processing_config() 
            with config_tabs[1]:
                app_instance.render_retrieval_config() 
            with config_tabs[2]:
                app_instance.render_evaluation_config() 
                app_instance.render_evaluation_section() 

    @staticmethod
    def render_chat_area(app_instance):
        """
        Renders the main chat area.
        """
        st.subheader("Chat with your Documents")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                app_instance.process_chat_input(prompt)
