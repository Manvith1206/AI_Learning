"""
Session Manager - Handles Streamlit session state management
"""
import streamlit as st
from typing import List, Dict, Any, Optional
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class SessionManager:
    """Manager for Streamlit session state"""
    
    def __init__(self):
        """Initialize the session manager"""
        pass

    def initializeVectorizer(self):
        st.session_state.vectorizer = TfidfVectorizer()

    def initializeVectors(self):
        st.session_state.vectors = None

    def addVectors(self, vectorDocs):
        st.session_state.vectors = vectorDocs
    
    def getVectorizer(self):
        if "vectorizer" in st.session_state:
            return st.session_state.vectorizer
    def initializeNN_Model(self, texts: str):
        st.session_state.nn_model = NearestNeighbors(
            n_neighbors=min(5, len(texts)),  # Limit to 5 or number of texts if less
            metric='cosine'
        )
        st.session_state.nn_model.fit(st.session_state.vectors)
        
    # Message management for chat interface
    def has_messages(self) -> bool:
        """Check if messages exist in session state"""
        return "messages" in st.session_state
    
    def initialize_messages(self) -> None:
        """Initialize empty messages list in session state"""
        st.session_state.messages = []
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages from session state"""
        if not self.has_messages():
            self.initialize_messages()
        return st.session_state.messages
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to session state
        
        Args:
            content: Message content
        """
        if not self.has_messages():
            self.initialize_messages()
        st.session_state.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to session state
        
        Args:
            content: Message content
        """
        if not self.has_messages():
            self.initialize_messages()
        st.session_state.messages.append({"role": "assistant", "content": content})
    
    def clear_messages(self) -> None:
        """Clear all messages from session state"""
        st.session_state.messages = []
    
    # Image management for image generator
    def has_images(self) -> bool:
        """Check if images exist in session state"""
        return "images" in st.session_state
    
    def initialize_images(self) -> None:
        """Initialize empty images list and latest image in session state"""
        st.session_state.images = []
        st.session_state.latest_image = None
    
    def get_images(self) -> List[Dict[str, Any]]:
        """Get all images from session state"""
        if not self.has_images():
            self.initialize_images()
        return st.session_state.images
    
    def get_latest_image(self) -> Optional[Image.Image]:
        """Get the latest generated image from session state"""
        if "latest_image" not in st.session_state:
            return None
        return st.session_state.latest_image
    
    def add_user_image_prompt(self, content: str) -> None:
        """
        Add a user image prompt to session state
        
        Args:
            content: Prompt content
        """
        if not self.has_images():
            self.initialize_images()
        st.session_state.images.append({"role": "user", "content": content, "type": "text"})
    
    def add_assistant_image(self, image: Image.Image) -> None:
        """
        Add an assistant-generated image to session state
        
        Args:
            image: Generated image
        """
        if not self.has_images():
            self.initialize_images()
        st.session_state.images.append({"role": "assistant", "content": image, "type": "image"})
        st.session_state.latest_image = image
    
    def clear_images(self) -> None:
        """Clear all images from session state"""
        st.session_state.images = []
        st.session_state.latest_image = None
    
    # Model management
    def set_model(self, model_name: str) -> None:
        """
        Set the current model in session state
        
        Args:
            model_name: Name of the model
        """
        st.session_state.gemini_model = model_name
    
    def get_model(self, default_model: str) -> str:
        """
        Get the current model from session state
        
        Args:
            default_model: Default model to use if not set
            
        Returns:
            Current model name
        """
        if "gemini_model" not in st.session_state:
            st.session_state.gemini_model = default_model
        return st.session_state.gemini_model
    
    def set_prompt(self, prompt: str):
        if not "prompt" in st.session_state:
            st.session_state.prompt = ""
        st.session_state.prompt = prompt

    def get_prompt(self):
        if "prompt" in st.session_state:
            return st.session_state.prompt
        
    def set_docs(self, docs: list):
        if not "docs" in st.session_state:
            st.session_state.docs = []
        st.session_state.docs = docs

    def get_docs(self):
        if "docs" in st.session_state:
            return st.session_state.docs