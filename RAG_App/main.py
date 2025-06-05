import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add parent directory to sys.path for module resolution
import pandas as pd
from typing import Dict, List, Any, Optional
from UI.pages.main_page import MainPage
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Initialize the main page of the application
    main_page = MainPage()
    main_page.render()

if __name__ == "__main__":
    main()
