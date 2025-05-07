from .base_llm_service import BaseLLMService
import cohere

class CohereChat(BaseLLMService):
    def __init__(self, apikey, model_name):
        self.client = cohere.client_v2(apikey)
        self.model_name = model_name

    def generate_response(self, prompt, **kwargs):
        response = self.client.chat(
            model=self.model_name,
            messages=prompt
        )
        return response