from UI.UI_Components import UIComponents
# Sample questions and answers
flashcards = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    {"question": "What is the boiling point of water?", "answer": "100°C or 212°F"},
]

class Flashcard:
    def __init__(self, flashcards):
        self.flashcards = flashcards
        self.current_index = 0
        self.show_answer = False
        self.card = None
                
        # Streamlit App
        UIComponents.create_title("📚 Flashcards")

    def initialze_session_state(self):
        UIComponents.get_session_state_variable("card_index", 0)
        UIComponents.get_session_state_variable("show_answer", False)
        # if "card_index" not in st.session_state:
        #     st.session_state.card_index = 0
        # if "show_answer" not in st.session_state:
        #     st.session_state.show_answer = False

    def create_columns(self):        
        # Navigation buttons
        col1, col2, col3 = UIComponents.create_columns([1, 1, 2])

        with col1:
            if UIComponents.create_button("⬅️ Previous"):
                if UIComponents.get_session_state_variable("card_index", 0) > 0:
                    card_index = UIComponents.get_session_state_variable("card_index", 0)
                    card_index -= 1
                    UIComponents.set_session_state_variable("card_index", card_index)
                    UIComponents.set_session_state_variable("show_answer", False)

        with col2:
            if UIComponents.create_button("➡️ Next"):
                if UIComponents.get_session_state_variable("card_index", 0) < len(flashcards) - 1:
                    card_index = UIComponents.get_session_state_variable("card_index", 0)
                    card_index += 1
                    UIComponents.set_session_state_variable("card_index", card_index)
                    UIComponents.set_session_state_variable("show_answer", False)

    def display_card(self):        
        # Current flashcard
        card = flashcards[UIComponents.set_session_state_variable("card_index", 0)]
        UIComponents.create_subheader_UI(f"Question {UIComponents.set_session_state_variable("card_index", 0) + 1} of {len(flashcards)}")
        UIComponents.markdown(f"**Q:** {card['question']}")
        self.card = card

    def show_or_hide_answer(self):        
        # Show/Hide answer
        if UIComponents.create_button("Show Answer" if not UIComponents.get_session_state_messages("show_answer", False) else "Hide Answer"):
            UIComponents.set_session_state_variable("show_answer", False) = not UIComponents.set_session_state_variable("show_answer", False)

        if UIComponents.set_session_state_variable("show_answer", False):
            UIComponents.markdown(f"**A:** {self.card['answer']}")
