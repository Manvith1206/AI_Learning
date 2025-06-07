import os
import sys

from UI.UI_Components import UIComponents
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add parent directory to sys.path for module resolution
import pandas as pd
from typing import Dict, List, Any, Optional
from UI.pages.main_page import MainPage
from config import ConfigManager
import Utils.Utils as Utils
from dotenv import load_dotenv

def main():
    load_dotenv()
    config_manager = ConfigManager()
    UIComponents.initialize_pipeline(ConfigManager())
    
    main_page = MainPage(Utils.get_pipeline(), config_manager)
    main_page.render()

if __name__ == "__main__":
    main()
