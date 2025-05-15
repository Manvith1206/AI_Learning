from .base_llm_service import BaseLLMService
from rag_modular.RAG_Constants import GeminiLLMModel
from google.genai import types

class GeminiService(BaseLLMService):
    def __init__(self, client, model_name=GeminiLLMModel.GEMINI_FLASH.value):
        self.client = client
        self.model_name = model_name
    def generate_response(self, prompt, **kwargs):
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
