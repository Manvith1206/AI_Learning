import os
import sys

from UI.UI_Components import UIComponents
from infrastructure.Common.rag_pipeline import RAGPipeline
from config import ConfigManager
from UI.pages.main_page import MainPage
import Utils.Utils as Utils
from dotenv import load_dotenv
def main():
    """Main function to run the RAG Application."""
    UIComponents.initialize_page()
    config_manager = ConfigManager()
    UIComponents.get_session_state_variable("pipeline_config", config_manager)
    # This is where the pipeline initialization logic will be updated.
    # For now, we keep the existing structure.
    pipeline = Utils.get_pipeline()
    
    main_page = MainPage(pipeline=pipeline, config_manager=config_manager)
    main_page.render()

if __name__ == "__main__":
    load_dotenv()
    main()
