from sklearn.metrics.pairwise import cosine_similarity
from infrastructure.LLM_Chat_Services.base_llm_service import BaseLLMService
from infrastructure.Embedders.base_embedder import BaseEmbedder

class LLM_Evaluation_Service():
    """Protocol for a generic LLM service."""
    def __init__(self, client: BaseLLMService, model_name: str, embedder: BaseEmbedder):
        self.client = client
        self.model_name = model_name
        self.embedder = embedder

    def evaluate_statement(self, statement: str, context: str, prompt_template: str):
        """Evaluates if a statement is supported by the given context using a specific prompt."""
        prompt = prompt_template.format(statement=statement, context=context)
        try:
            response = self.client.generate_response(prompt)
            decision = response
            return "yes" in decision
        except Exception as e:
            print(f"Error during Gemini statement evaluation: {e}")
            return False # Default to false on error

    def generate_text(self, prompt: str):
        """Generates text based on a given prompt."""
        try:
            response = self.client.generate_response(prompt)
            return response
        except Exception as e:
            print(f"Error during Gemini text generation: {e}")
            return "" # Default to empty string on error

    def generate_questions(self, answer: str, original_question: str, prompt_template: str, num_questions: int = 3):
        """Generates questions for which the given answer would be appropriate."""
        prompt = prompt_template.format(answer=answer, num_questions=num_questions, original_question=original_question)
        try:
            response_text = self.generate_text(prompt)
            # Assuming the LLM returns questions separated by newlines
            questions = [q.strip() for q in response_text.split('\n') if q.strip()]
            return questions[:num_questions]
        except Exception as e:
            print(f"Error during Gemini question generation: {e}")
            return []
        
    def calculate_similarity(self, text1: str, text2: str):
        """Calculates semantic similarity between two texts (e.g., using embeddings and cosine similarity)."""
        try:
            breakpoint()
            text1_vec = self.embedder.transform([text1])  # shape (1, dim)
            text2_vecs = self.embedder.transform(text2)  # shape (n, dim)

            sims = cosine_similarity(text1_vec, text2_vecs).flatten()
            print("Similarity: ", float(sims))

            return float(sims)
        except Exception as e:
            print(f"Error during Gemini similarity calculation: {e}")
            return 0.0 # Default to 0.0 on error
        