import time
from .base_llm_service import BaseLLMService
import anthropic

class ClaudeService(BaseLLMService):
    def __init__(self, client, model_name="claude-2"):
        self.client = client
        self.model_name = model_name
        self.time_taken = 0
        self.cost = 0
    def generate_response(self, prompt, **kwargs):
        start_time = time.time()
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        self.time_taken = end_time - start_time

        
        return response.content[0].text
    def get_cost_and_time_taken(self):
        """
        Get the cost and time taken for the Claude service call
        """
        return self.cost, self.time_taken