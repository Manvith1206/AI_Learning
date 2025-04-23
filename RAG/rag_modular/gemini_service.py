from .base_llm_service import BaseLLMService

class GeminiService(BaseLLMService):
    def __init__(self, client, model_name="gemini-2.0-flash"):
        self.client = client
        self.model_name = model_name
    def generate_response(self, prompt, **kwargs):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
