"""
Main Streamlit Application Entry Point
"""
import streamlit as st
from PIL import Image
from io import BytesIO

# Import services
from services.gemini_service import GeminiService
from services.document_service import DocumentService
from services.rag_service import RagService

# Import UI components
from ui.chat_interface import ChatInterface
from ui.image_editor import ImageEditor
from ui.image_generator import ImageGenerator

# Import utilities
from utils.constants import (
    APP_TITLE, TAB_CHAT, TAB_EDIT_IMAGE, TAB_GENERATE_IMAGE, 
    MODEL_CHAT, MODEL_IMAGE_GENERATION, MODEL_IMAGE_EDITING,
    CONTEXT_CHAT, CONTEXT_MAIN, CONTEXT_IMAGE_GENERATION,
    CSS_CHAT_INPUT
)
from utils.session_manager import SessionManager

# Set page title
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Apply custom CSS
st.markdown(CSS_CHAT_INPUT, unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def initialize_services():
    """Initialize and cache services"""
    gemini_service = GeminiService(api_key=st.secrets["GEMINI_API_KEY"])
    document_service = DocumentService()
    
    return gemini_service, document_service, rag_service

# Initialize session manager
session_manager = SessionManager()

# Initialize services
gemini_service, document_service, rag_service = initialize_services()

# Create tabs
chat_tab, edit_image_tab, generate_image_tab = st.tabs([
    TAB_CHAT, 
    TAB_EDIT_IMAGE, 
    TAB_GENERATE_IMAGE
])

# Set up chat context
chat_context = CONTEXT_CHAT.format(CONTEXT_MAIN)

# Render UI components in their respective tabs
with chat_tab:
    chat_interface = ChatInterface(
        gemini_service=gemini_service,
        session_manager=session_manager,
        model_name=MODEL_CHAT,
        context=chat_context
    )
    chat_interface.render()

with edit_image_tab:
    image_editor = ImageEditor(
        gemini_service=gemini_service,
        model_name=MODEL_IMAGE_EDITING
    )
    image_editor.render()

with generate_image_tab:
    image_generator = ImageGenerator(
        gemini_service=gemini_service,
        session_manager=session_manager,
        model_name=MODEL_IMAGE_GENERATION,
        context=CONTEXT_IMAGE_GENERATION
    )
    image_generator.render()
