import streamlit as st
from dotenv import load_dotenv
from config import ConfigManager
from UI.pages.main_page import MainPage
from infrastructure.common.pipelines.rag_pipeline import RAGPipeline
from UI.UI_Components import UIComponents
import infrastructure.common.RAG_Constants as constants
from infrastructure.common.exceptions import ComponentBuildError, MissingConfigurationError

def initialize_pipeline():
    """
    Loads configuration and initializes the RAG pipeline.
    Handles errors during initialization and stores the pipeline in session state.
    """
    if 'rag_pipeline' not in st.session_state:
        try:
            with UIComponents.display_spinner(constants.LOADING_DISPLAY_MESSAGE_FOR_INITIALZING_PAGE):
                load_dotenv()
                app_config = ConfigManager.load_config()
                st.session_state.rag_pipeline = RAGPipeline(app_config)
                st.session_state.error = None  # Clear any previous errors
        except (ComponentBuildError, MissingConfigurationError) as e:
            st.session_state.rag_pipeline = None
            st.session_state.error = str(e)
        except Exception as e:
            st.session_state.rag_pipeline = None
            st.session_state.error = f"An unexpected error occurred during initialization: {e}"

def main():
    """Main function to run the RAG Application."""
    UIComponents.initialize_page()
    initialize_pipeline()

    # If an error occurred during initialization, display it and stop.
    if st.session_state.get('error'):
        UIComponents.display_error(st.session_state.error)
        return

    # If pipeline is not available for other reasons, show a message.
    if not st.session_state.get('rag_pipeline'):
        st.warning("Pipeline could not be initialized. Please check the configuration and logs.")
        return

    # Retrieve the pipeline and config from session state.
    rag_pipeline = st.session_state.rag_pipeline
    app_config = rag_pipeline.config

    # Render the main page.
    with UIComponents.display_spinner(constants.LOADING_DISPLAY_MESSAGE_FOR_MAIN_PAGE):
        main_page = MainPage(pipeline=rag_pipeline, config=app_config)
        main_page.render()

if __name__ == "__main__":
    main()
