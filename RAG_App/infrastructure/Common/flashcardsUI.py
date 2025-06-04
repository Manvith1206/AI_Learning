import streamlit as st

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
        st.title("📚 Flashcards")

    def initialze_session_state(self):
        if "card_index" not in st.session_state:
            st.session_state.card_index = 0
        if "show_answer" not in st.session_state:
            st.session_state.show_answer = False

    def create_columns(self):        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("⬅️ Previous"):
                if st.session_state.card_index > 0:
                    st.session_state.card_index -= 1
                    st.session_state.show_answer = False

        with col2:
            if st.button("➡️ Next"):
                if st.session_state.card_index < len(flashcards) - 1:
                    st.session_state.card_index += 1
                    st.session_state.show_answer = False

    def display_card(self):        
        # Current flashcard
        card = flashcards[st.session_state.card_index]
        st.subheader(f"Question {st.session_state.card_index + 1} of {len(flashcards)}")
        st.markdown(f"**Q:** {card['question']}")
        self.card = card

    def show_or_hide_answer(self):        
        # Show/Hide answer
        if st.button("Show Answer" if not st.session_state.show_answer else "Hide Answer"):
            st.session_state.show_answer = not st.session_state.show_answer

        if st.session_state.show_answer:
            st.markdown(f"**A:** {self.card['answer']}")
if __name__ == "__main__":
    flashcard = Flashcard(flashcards)
    flashcard.initialze_session_state()
    
    # Create columns for navigation
    flashcard.create_columns()
    
    # Display current flashcard
    flashcard.display_card()
    
    # Show or hide answer
    flashcard.show_or_hide_answer()