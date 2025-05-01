from .base_llm_service import BaseLLMService
import anthropic

class ClaudeService(BaseLLMService):
    def __init__(self, client, model_name="claude-2"):
        self.client = client
        self.model_name = model_name
    def generate_response(self, prompt, **kwargs):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        
        return response.content[0].text