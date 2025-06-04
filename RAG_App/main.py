import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add parent directory to sys.path for module resolution
import pandas as pd
from typing import Dict, List, Any, Optional
from UI.pages.main_page import MainPage

def main():
    main_page = MainPage()
    main_page.render()

if __name__ == "__main__":
    main()
