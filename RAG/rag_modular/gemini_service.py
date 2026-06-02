from .base_llm_service import BaseLLMService
from rag_modular.RAG_Constants import GeminiLLMModel

class GeminiService(BaseLLMService):
    def __init__(self, client, model_name=GeminiLLMModel.GEMINI_FLASH.value):
        self.client = client
        self.model_name = model_name
    def generate_response(self, prompt, **kwargs):
        model_name = kwargs.get("model_name", self.model_name)
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
