from typing import List
from UI.UI_Components import UIComponents

class FlashcardDisplay:
    def __init__(self, flashcards: List[dict]):
        """
        Initializes the FlashcardDisplay component.

        Args:
            flashcards (List[dict]): A list of flashcard dictionaries, each with 'question' and 'answer'.
        """
        self.flashcards = flashcards

    def initialize_session_state(self):
        """Properly initialize session state variables if not already set."""
        if "card_index" not in UIComponents.get_session_state():
            UIComponents.set_session_state_variable("card_index", 0)
        if "show_answer" not in UIComponents.get_session_state():
            UIComponents.set_session_state_variable("show_answer", False)

    def create_columns(self):
        """Creates navigation buttons for the flashcards."""
        if not self.flashcards:
            return
        
        col1, col2, _ = UIComponents.create_columns([1, 1, 2])

        with col1:
            if UIComponents.create_button("⬅️ Previous"):
                card_index = UIComponents.get_session_state_variable("card_index", 0)
                if card_index > 0:
                    UIComponents.set_session_state_variable("card_index", card_index - 1)
                    UIComponents.set_session_state_variable("show_answer", False)
                    UIComponents.rerun()

        with col2:
            if UIComponents.create_button("➡️ Next"):
                card_index = UIComponents.get_session_state_variable("card_index", 0)
                if card_index < len(self.flashcards) - 1:
                    UIComponents.set_session_state_variable("card_index", card_index + 1)
                    UIComponents.set_session_state_variable("show_answer", False)
                    UIComponents.rerun()

    def display_card(self):
        """Displays the current flashcard's question."""
        if not self.flashcards:
            return

        card_index = UIComponents.get_session_state_variable("card_index", 0)
        
        if not 0 <= card_index < len(self.flashcards):
            UIComponents.display_warning("Flashcard index is out of range. Resetting.")
            UIComponents.set_session_state_variable("card_index", 0)
            card_index = 0

        card = self.flashcards[card_index]
        UIComponents.create_subheader_UI(f"Question {card_index + 1} of {len(self.flashcards)}")
        UIComponents.markdown(f"**Q:** {card['question']}")

    def show_or_hide_answer(self):
        """Shows or hides the answer for the current flashcard."""
        show_answer = UIComponents.get_session_state_variable("show_answer", False)
        button_label = "Hide Answer" if show_answer else "Show Answer"
        
        if UIComponents.create_button(button_label):
            UIComponents.set_session_state_variable("show_answer", not show_answer)
            UIComponents.rerun()
        
        if UIComponents.get_session_state_variable("show_answer", False):
            card_index = UIComponents.get_session_state_variable("card_index", 0)
            if self.flashcards and 0 <= card_index < len(self.flashcards):
                card = self.flashcards[card_index]
                UIComponents.markdown(f"**A:** {card['answer']}")
