import streamlit as st
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add parent directory to sys.path for module resolution
import pandas as pd
from typing import Dict, List, Any, Optional
from RAG_App.UI.sidebar import Sidebar

def main():
    Sidebar().run()

if __name__ == "__main__":
    main()
