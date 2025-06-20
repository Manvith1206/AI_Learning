import os
import sys

from UI.ui_components import UIComponents
from config import ConfigManager
from UI.pages.main_page import MainPage
import Utils.utils as Utils
import infrastructure.common.rag_constants as constants

from dotenv import load_dotenv

def main():
    """Main function to run the RAG Application."""
    UIComponents.initialize_page()
    with UIComponents.display_spinner(constants.LOADING_DISPLAY_MESSAGE_FOR_INITIALZING_PAGE):
        config_manager = ConfigManager()
        UIComponents.get_session_state_variable("pipeline_config", config_manager)

    # This is where the pipeline initialization logic will be updated.
    # For now, we keep the existing structure.
    pipeline = Utils.get_pipeline()
    
    with UIComponents.display_spinner(constants.LOADING_DISPLAY_MESSAGE_FOR_MAIN_PAGE):
        main_page = MainPage(pipeline=pipeline, config_manager=config_manager)
        main_page.render()

if __name__ == "__main__":
    load_dotenv()
    main()
