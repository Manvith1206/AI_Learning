import json
from typing import List, Dict
import traceback

class FlashCardGeneration:
    def __init__(self):
        pass
    def generate_flashcards_from_text(self, text_content: str, num_flashcards: int = 5) -> List[Dict[str, str]]:
        """Generates flashcards from the given text content using the LLM service."""
        if not text_content.strip():
            self.warning_callback("Cannot generate flashcards from empty content.")
            return []

        prompt = self.flashcard_prompt_provider.get_final_prompt(text_content=text_content, num_flashcards=num_flashcards)
        
        try:
            full_response = ""
            # Assuming llm_service.generate_response is a generator yielding response chunks
            for delta in self.llm_service.generate_response(prompt):
                full_response += delta
            
            # Attempt to parse the LLM's response as JSON
            # The response might be wrapped in markdown code blocks, try to strip them
            if full_response.strip().startswith("```json"):
                full_response = full_response.strip()[7:-3].strip()
            elif full_response.strip().startswith("```"):
                 full_response = full_response.strip()[3:-3].strip()

            flashcards = json.loads(full_response)
            
            # Validate structure
            if not isinstance(flashcards, list):
                raise ValueError("LLM response is not a list.")
            for card in flashcards:
                if not (isinstance(card, dict) and "question" in card and "answer" in card):
                    raise ValueError("Invalid flashcard structure in LLM response.")
            
            return flashcards[:num_flashcards] # Return up to the requested number

        except json.JSONDecodeError as e:
            self.error_callback(f"Error decoding JSON from LLM for flashcards: {e}\nRaw response: {full_response}")
            print(f"JSONDecodeError: {e}. Raw LLM response for flashcards:\n{full_response}")
            return []
        except ValueError as e:
            self.error_callback(f"Error in flashcard data structure from LLM: {e}\nRaw response: {full_response}")
            print(f"ValueError: {e}. Raw LLM response for flashcards:\n{full_response}")
            return []
        except Exception as e:
            self.error_callback(f"An unexpected error occurred during flashcard generation: {e}")
            print(f"Unexpected error in generate_flashcards_from_text: {e}, Traceback: {traceback.format_exc()}")
            return []
