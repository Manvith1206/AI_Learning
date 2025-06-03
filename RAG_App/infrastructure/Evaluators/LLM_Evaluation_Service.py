class LLM_Evaluation_Service():
    """Protocol for a generic LLM service."""

    def evaluate_statement(self, statement: str, context: str, prompt_template: str):
        """Evaluates if a statement is supported by the given context using a specific prompt."""
        ...

    def generate_text(self, prompt: str):
        """Generates text based on a given prompt."""
        ...

    def generate_questions(self, answer: str, prompt_template: str, num_questions: int = 3):
        """Generates questions for which the given answer would be appropriate."""
        ...

    def calculate_similarity(self, text1: str, text2: str):
        """Calculates semantic similarity between two texts (e.g., using embeddings and cosine similarity)."""
        # This might involve calling an embedding model and then a similarity function.
        # For simplicity, this might be a direct call if the LLM service supports it,
        # or it might need a separate embedding component.
        ...