import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from typing import List, Dict, Any, Callable
from UI.UI_Components import UIComponents
import time

class FlashcardDisplay:
    def __init__(self, flashcards):
        self.flashcards = flashcards
        self.current_index = 0
        self.show_answer = False
        self.card = None

    def initialze_session_state(self):
        # Properly initialize session state variables if not already set
        if "card_index" not in UIComponents.get_session_state():
            UIComponents.set_session_state_variable("card_index", 0)
        if "show_answer" not in UIComponents.get_session_state():
            UIComponents.set_session_state_variable("show_answer", False)

    def create_columns(self):        
        if not self.flashcards or not self.flashcards:
            return
        # Navigation buttons
        col1, col2, col3 = UIComponents.create_columns([1, 1, 2])

        with col1:
            if UIComponents.create_button("⬅️ Previous"):
                card_index = UIComponents.get_session_state_variable("card_index", 0)
                if card_index > 0:
                    card_index -= 1
                    UIComponents.set_session_state_variable("card_index", card_index)
                    UIComponents.set_session_state_variable("show_answer", False)
                    print(f"Previous Button / Card index updated to: {card_index}")
                    # Removed self.display_card() to avoid duplicate UI

        with col2:
            if UIComponents.create_button("➡️ Next"):
                card_index = UIComponents.get_session_state_variable("card_index", 0)
                if card_index < len(self.flashcards) - 1:
                    card_index += 1
                    UIComponents.set_session_state_variable("card_index", card_index)
                    UIComponents.set_session_state_variable("show_answer", False)
                    print(f"Next Button / Card index updated to: {card_index}")
                    # Removed self.display_card() to avoid duplicate UI

    def display_card(self):        
        if not self.flashcards:
            return

        card_index = UIComponents.get_session_state_variable("card_index", 0)
        print(f"Displaying card at index: {card_index}")

        # Current flashcard
        card = self.flashcards[card_index]
        UIComponents.create_subheader_UI(f"Question {card_index + 1} of {len(self.flashcards)}")
        UIComponents.markdown(f"**Q:** {card['question']}")
        self.card = card

    def show_or_hide_answer(self):        
        show_answer = UIComponents.get_session_state_variable("show_answer", False)
        button_label = "Hide Answer" if show_answer else "Show Answer"
        if UIComponents.create_button(button_label):
            show_answer = not show_answer
            UIComponents.set_session_state_variable("show_answer", show_answer)
            # Optionally, re-render the card to update the button label immediately
            # self.display_card()  # Uncomment if needed for your UI framework
        if show_answer and self.card:
            UIComponents.markdown(f"**A:** {self.card['answer']}")
