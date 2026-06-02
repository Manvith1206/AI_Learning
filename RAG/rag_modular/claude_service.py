from .base_llm_service import BaseLLMService
import anthropic

class ClaudeService(BaseLLMService):
    def __init__(self, client, model_name="claude-2"):
        self.client = client
        self.model_name = model_name
    def generate_response(self, prompt, **kwargs):
        model_name = kwargs.get("model_name", self.model_name)
        response = self.client.messages.create(
            model=model_name,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        
        return response.content[0].text
