from typing import List, Dict, Any, Callable
from models import Flashcard
from UI.UI_Components import UIComponents


class FlashcardDisplay:
    """Component for displaying and generating flashcards"""
    
    def __init__(self, on_generate_flashcards: Callable[[], List[Flashcard]]):
        """
        Initialize the flashcard display component
        
        Args:
            on_generate_flashcards: Callback function to generate flashcards
        """
        self.on_generate_flashcards = on_generate_flashcards
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for flashcards"""
        UIComponents.initialize_session_state({
            "flashcards": [],
            "flashcards_generation_attempted": False
        })
    
    def render(self):
        """Render the flashcard interface"""
        UIComponents.create_subheader_UI.subheader("Flashcards")
        
        # Display info about flashcards
        UIComponents.write("""
        Flashcards are automatically generated from your documents to help you learn and review key concepts.
        Click the button below to generate flashcards from your uploaded documents.
        """)
        
        # Generate flashcards button
        self._render_generation_button()
        
        # Display flashcards
        self._render_flashcards()
    
    def _render_generation_button(self):
        """Render the flashcard generation button"""
        col1, col2 = UIComponents.create_columns([1, 3])
        
        with col1:
            if UIComponents.create_button("Generate Flashcards", key="generate_flashcards"):
                self._trigger_flashcard_generation()
    
    def _trigger_flashcard_generation(self):
        """Handle flashcard generation"""
        UIComponents.set_session_state_variable("flashcards_generation_attempted", False)
        
        if not UIComponents.get_session_state_variable("documents", None):
            UIComponents.display_error("Please upload and process documents first.")
            return
        
        with UIComponents.display_spinner("Generating flashcards..."):
            try:
                flashcards = self.on_generate_flashcards()
                UIComponents.set_session_state_variable("flashcards", flashcards)
                UIComponents.display_success(f"Generated {len(flashcards)} flashcards!")
            except Exception as e:
                UIComponents.display_error(f"Error generating flashcards: {str(e)}")
    
    def _render_flashcards(self):
        """Render the generated flashcards"""
        if not UIComponents.get_session_state_variable("flashcards", None) and UIComponents.get_session_state_variable("flashcards_generation_attempted", None):
            UIComponents.display_info("No flashcards have been generated yet.")
            return
        
        for i, flashcard in enumerate(UIComponents.get_session_state_variable("flashcards", [])):
            with UIComponents.create_expander(f"Flashcard {i+1}: {flashcard.question[:50]}...", expanded=False):
                UIComponents.create_subheader_UI("**Question:**")
                UIComponents.create_subheader_UI(flashcard.question)
                UIComponents.create_subheader_UI("**Answer:**")
                UIComponents.create_subheader_UI(flashcard.answer)
