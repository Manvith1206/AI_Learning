import os
import sys
from tkinter import constants
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add parent directory to sys.path for module resolution
import pandas as pd
from typing import Dict, List, Any, Optional
from UI.pages.main_page import MainPage
from dotenv import load_dotenv
from config import ConfigManager
from infrastructure.Common.rag_pipeline import RAGPipeline
from infrastructure.Common.exceptions import ComponentBuildError, MissingConfigurationError, InvalidConfigurationError, PipelineError # Updated imports
import logging
import streamlit as st # For displaying critical errors if UIComponents isn't ready
import infrastructure.Common.RAG_Constants as constants
def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration manager
        config_manager = ConfigManager()

        # Load API Keys from environment variables
        logger.info("Loading API keys from environment variables...")
        gemini_api_key = config_manager.get_secret(constants.GEMINI_API_KEY)
        anthropic_api_key = config_manager.get_secret(constants.CLAUDE_API_KEY)
        cohere_api_key = config_manager.get_secret(constants.COHERE_API_KEY)
        openai_api_key = config_manager.get_secret(constants.OPENAI_API_KEY)
        pinecone_api_key = config_manager.get_secret(constants.PINECONE_API_KEY)
        voyage_api_key = config_manager.get_secret(constants.VOYAGE_API_KEY)
        mistral_api_key = config_manager.get_secret(constants.MISTRAL_API_KEY)
        jina_api_key = config_manager.get_secret(constants.JINA_API_KEY)

        # Log loaded keys (optional, for debugging - be careful with actual key values in logs)
        # logger.debug(f"Gemini Key Loaded: {'Yes' if gemini_api_key else 'No'}")
        # ... similar for other keys

        # Initialize RAG Pipeline
        logger.info("Initializing RAG Pipeline...")
        pipeline = RAGPipeline(
            config_manager=config_manager,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            cohere_api_key=cohere_api_key,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            voyage_api_key=voyage_api_key,
            mistral_api_key=mistral_api_key,
            jina_api_key=jina_api_key
        )
        logger.info("RAG Pipeline initialized successfully.")

    except MissingConfigurationError as e:
        logger.error(f"CRITICAL ERROR: Missing configuration for RAG Pipeline: {e}")
        st.error(f"Application Critical Error: Missing essential configuration. Please check your API keys and config files. Details: {e}")
        return # Stop the app if pipeline can't be created
    except ComponentBuildError as e:
        logger.error(f"CRITICAL ERROR: Failed to build a component in RAG Pipeline: {e}")
        st.error(f"Application Critical Error: Failed to initialize a RAG component. Please check logs and component configurations. Details: {e}")
        return # Stop the app
    except InvalidConfigurationError as e:
        logger.error(f"CRITICAL ERROR: Invalid configuration for RAG Pipeline: {e}")
        st.error(f"Application Critical Error: Invalid configuration detected. Please check your config files. Details: {e}")
        return # Stop the app
    except PipelineError as e:
        logger.error(f"CRITICAL ERROR: A pipeline-level error occurred during RAG Pipeline initialization: {e}", exc_info=True)
        st.error(f"Application Critical Error: A pipeline error occurred during setup. Please check logs. Details: {e}")
        return # Stop the app
    except Exception as e:
        logger.error(f"CRITICAL ERROR: An unexpected error occurred during RAG Pipeline initialization: {e}", exc_info=True)
        st.error(f"Application Critical Error: An unexpected error occurred during setup. Please check logs. Details: {e}")
        return # Stop the app

    # Initialize the main page of the application, passing the pipeline instance
    logger.info("Initializing Main Page...")
    main_page = MainPage(pipeline=pipeline, config_manager=config_manager) # Pass config_manager too if MainPage needs it
    main_page.render()
    logger.info("Main Page rendered.")

if __name__ == "__main__":
    main()
