import os
import sys

from UI.UI_Components import UIComponents
from infrastructure.Common.rag_pipeline import RAGPipeline
from config import ConfigManager
from UI.pages.main_page import MainPage

def main():
    """Main function to run the RAG Application."""
    UIComponents.initialize_page()
    config_manager = ConfigManager(config_path='config.json')
    
    # This is where the pipeline initialization logic will be updated.
    # For now, we keep the existing structure.
    pipeline = RAGPipeline(
        config_manager=config_manager,
        warning_callback=UIComponents.display_warning,
        error_callback=UIComponents.display_error
    )
    
    main_page = MainPage(pipeline=pipeline, config_manager=config_manager)
    main_page.render()

if __name__ == "__main__":
    main()
